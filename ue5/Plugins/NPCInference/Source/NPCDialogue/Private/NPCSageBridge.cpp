// NPCSageBridge.cpp
// UE5 <-> Python bridge for SAGE behavior-tree tasks.

#include "NPCSageBridge.h"

#include "Containers/StringConv.h"
#include "Dom/JsonObject.h"
#include "HAL/CriticalSection.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformMisc.h"
#include "HAL/PlatformProcess.h"
#include "Misc/Base64.h"
#include "Misc/FileHelper.h"
#include "Misc/Guid.h"
#include "Misc/Paths.h"
#include "Misc/ScopeLock.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

namespace
{
FCriticalSection GDaemonMutex;
FProcHandle GDaemonProc;
FString GDaemonShutdownFilePath;

const TCHAR* GDaemonScriptFile = TEXT("sage_bt_daemon.py");

constexpr double GDefaultResponseTimeoutSeconds = 6.0;
constexpr double GDefaultPollMs = 8.0;
constexpr double GDefaultIdleSleepMs = 16.0;
constexpr double GDefaultBackoffMs = 30.0;
constexpr int32 GDefaultMaxRetries = 1;
constexpr double GDefaultStaleAgeSeconds = 180.0;

FString ToBase64Utf8(const FString& Raw)
{
    FTCHARToUTF8 Converter(*Raw);
    return FBase64::Encode(reinterpret_cast<const uint8*>(Converter.Get()), Converter.Length());
}

double ReadEnvDouble(const TCHAR* Name, double DefaultValue, double MinValue, double MaxValue)
{
    const FString Raw = FPlatformMisc::GetEnvironmentVariable(Name);
    if (Raw.IsEmpty())
    {
        return DefaultValue;
    }

    const double Parsed = FCString::Atod(*Raw);
    if (!FMath::IsFinite(Parsed))
    {
        return DefaultValue;
    }
    return FMath::Clamp(Parsed, MinValue, MaxValue);
}

int32 ReadEnvInt(const TCHAR* Name, int32 DefaultValue, int32 MinValue, int32 MaxValue)
{
    const FString Raw = FPlatformMisc::GetEnvironmentVariable(Name);
    if (Raw.IsEmpty())
    {
        return DefaultValue;
    }

    const int32 Parsed = FCString::Atoi(*Raw);
    return FMath::Clamp(Parsed, MinValue, MaxValue);
}

double BridgeTimeoutSeconds()
{
    return ReadEnvDouble(TEXT("SAGE_BRIDGE_TIMEOUT_SEC"), GDefaultResponseTimeoutSeconds, 0.5, 120.0);
}

double BridgePollSeconds()
{
    return ReadEnvDouble(TEXT("SAGE_BRIDGE_POLL_MS"), GDefaultPollMs, 1.0, 500.0) / 1000.0;
}

double BridgeIdleSleepSeconds()
{
    return ReadEnvDouble(TEXT("SAGE_BRIDGE_IDLE_SLEEP_MS"), GDefaultIdleSleepMs, 1.0, 1000.0) / 1000.0;
}

double BridgeRetryBackoffSeconds()
{
    return ReadEnvDouble(TEXT("SAGE_BRIDGE_RETRY_BACKOFF_MS"), GDefaultBackoffMs, 1.0, 4000.0) / 1000.0;
}

int32 BridgeMaxRetries()
{
    return ReadEnvInt(TEXT("SAGE_BRIDGE_MAX_RETRIES"), GDefaultMaxRetries, 0, 5);
}

double BridgeStaleAgeSeconds()
{
    return ReadEnvDouble(TEXT("SAGE_BRIDGE_STALE_SEC"), GDefaultStaleAgeSeconds, 10.0, 86400.0);
}

FString QuoteArg(const FString& Raw)
{
    FString Escaped = Raw;
    Escaped.ReplaceInline(TEXT("\""), TEXT("\\\""));
    return FString::Printf(TEXT("\"%s\""), *Escaped);
}

FString BridgeRuntimeDir()
{
    return FPaths::ConvertRelativePathToFull(FPaths::ProjectDir(), TEXT("storage/runtime/sage_bridge"));
}

FString BridgeRequestDir()
{
    return FPaths::Combine(BridgeRuntimeDir(), TEXT("requests"));
}

FString BridgeResponseDir()
{
    return FPaths::Combine(BridgeRuntimeDir(), TEXT("responses"));
}

bool EnsureDirectory(const FString& Dir)
{
    return IFileManager::Get().MakeDirectory(*Dir, true);
}

void DeleteIfExists(const FString& FilePath)
{
    if (IFileManager::Get().FileExists(*FilePath))
    {
        IFileManager::Get().Delete(*FilePath, false, true);
    }
}

bool WriteFileAtomically(const FString& Path, const FString& Content)
{
    const FString TempPath = Path + TEXT(".tmp.") + FGuid::NewGuid().ToString(EGuidFormats::Digits);
    if (!FFileHelper::SaveStringToFile(Content, *TempPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
    {
        return false;
    }

    const bool bMoved = IFileManager::Get().Move(
        *Path,
        *TempPath,
        true,
        true,
        false,
        true
    );

    if (!bMoved)
    {
        DeleteIfExists(TempPath);
    }
    return bMoved;
}

void CleanupStaleFilesInDir(const FString& Dir, const FString& Pattern, double MaxAgeSeconds)
{
    if (!IFileManager::Get().DirectoryExists(*Dir))
    {
        return;
    }

    TArray<FString> Matches;
    IFileManager::Get().FindFiles(Matches, *FPaths::Combine(Dir, Pattern), true, false);

    const FDateTime Now = FDateTime::UtcNow();
    for (const FString& Name : Matches)
    {
        const FString FullPath = FPaths::Combine(Dir, Name);
        const FDateTime Stamp = IFileManager::Get().GetTimeStamp(*FullPath);
        if (Stamp == FDateTime::MinValue() || (Now - Stamp).GetTotalSeconds() > MaxAgeSeconds)
        {
            IFileManager::Get().Delete(*FullPath, false, true);
        }
    }
}

bool ParseDaemonResponse(
    const FString& ResponseJson,
    FString& OutStdOut,
    FString& OutStdErr,
    int32& OutReturnCode
)
{
    TSharedPtr<FJsonObject> Root;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ResponseJson);
    if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
    {
        return false;
    }

    Root->TryGetStringField(TEXT("stdout"), OutStdOut);
    Root->TryGetStringField(TEXT("stderr"), OutStdErr);

    double ReturnCode = -1.0;
    if (!Root->TryGetNumberField(TEXT("return_code"), ReturnCode))
    {
        return false;
    }
    OutReturnCode = static_cast<int32>(ReturnCode);
    return true;
}

FString SerializeJsonObject(const TSharedRef<FJsonObject>& Object)
{
    FString Json;
    TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Json);
    FJsonSerializer::Serialize(Object, Writer);
    return Json;
}

bool ParseJsonObject(const FString& JsonPayload, TSharedPtr<FJsonObject>& OutObject)
{
    OutObject.Reset();
    if (JsonPayload.IsEmpty())
    {
        return false;
    }

    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonPayload);
    return FJsonSerializer::Deserialize(Reader, OutObject) && OutObject.IsValid();
}

FString MakeFailurePayload(
    const FString& ErrorCode,
    const FString& StdErr,
    int32 ReturnCode,
    const FString& StdOut
)
{
    const TSharedRef<FJsonObject> Root = MakeShared<FJsonObject>();
    Root->SetBoolField(TEXT("ok"), false);
    Root->SetStringField(TEXT("error"), ErrorCode);
    Root->SetNumberField(TEXT("return_code"), ReturnCode);
    if (!StdErr.IsEmpty())
    {
        Root->SetStringField(TEXT("stderr"), StdErr);
    }
    if (!StdOut.IsEmpty())
    {
        Root->SetStringField(TEXT("stdout"), StdOut);
    }
    return SerializeJsonObject(Root);
}

FString AttachFailureMetadata(
    const FString& RawStdOut,
    const FString& ErrorCode,
    const FString& StdErr,
    int32 ReturnCode
)
{
    const FString Trimmed = RawStdOut.TrimStartAndEnd();
    if (!Trimmed.IsEmpty())
    {
        TSharedPtr<FJsonObject> Parsed;
        if (ParseJsonObject(Trimmed, Parsed) && Parsed.IsValid())
        {
            Parsed->SetBoolField(TEXT("ok"), false);
            if (!Parsed->HasField(TEXT("error")))
            {
                Parsed->SetStringField(TEXT("error"), ErrorCode);
            }
            Parsed->SetNumberField(TEXT("return_code"), ReturnCode);
            if (!StdErr.IsEmpty())
            {
                Parsed->SetStringField(TEXT("stderr"), StdErr);
            }
            return SerializeJsonObject(Parsed.ToSharedRef());
        }
    }

    return MakeFailurePayload(ErrorCode, StdErr, ReturnCode, Trimmed);
}

bool FinalizeCommandResult(
    const bool bExecuteOk,
    const FString& StdOut,
    const FString& StdErr,
    const int32 ReturnCode,
    FString& OutResultJson
)
{
    if (!bExecuteOk)
    {
        OutResultJson = MakeFailurePayload(TEXT("exec_failed"), StdErr, ReturnCode, StdOut);
        return false;
    }

    if (ReturnCode != 0)
    {
        OutResultJson = AttachFailureMetadata(StdOut, TEXT("handler_failed"), StdErr, ReturnCode);
        return false;
    }

    const FString Trimmed = StdOut.TrimStartAndEnd();
    OutResultJson = Trimmed.IsEmpty()
        ? FString(TEXT("{\"ok\":true}"))
        : Trimmed;
    return true;
}
} // namespace

void UNPCSageBridge::CleanupStaleIpcFiles()
{
    const FString RuntimeDir = BridgeRuntimeDir();
    const FString ReqDir = BridgeRequestDir();
    const FString RespDir = BridgeResponseDir();

    EnsureDirectory(RuntimeDir);
    EnsureDirectory(ReqDir);
    EnsureDirectory(RespDir);

    const double MaxAge = BridgeStaleAgeSeconds();
    CleanupStaleFilesInDir(ReqDir, TEXT("req_*.json"), MaxAge);
    CleanupStaleFilesInDir(RespDir, TEXT("resp_*.json"), MaxAge);
    CleanupStaleFilesInDir(ReqDir, TEXT("*.tmp*"), MaxAge);
    CleanupStaleFilesInDir(RespDir, TEXT("*.tmp*"), MaxAge);
}

FString UNPCSageBridge::ResolveHandlerScriptPath(const FString& HandlerScript)
{
    FString ScriptPath = HandlerScript;
    if (FPaths::IsRelative(ScriptPath))
    {
        ScriptPath = FPaths::ConvertRelativePathToFull(FPaths::ProjectDir(), ScriptPath);
    }
    return ScriptPath;
}

FString UNPCSageBridge::BuildCommandLineArgs(const TArray<FString>& Argv)
{
    FString Result;
    for (const FString& Arg : Argv)
    {
        if (!Result.IsEmpty())
        {
            Result += TEXT(" ");
        }
        Result += QuoteArg(Arg);
    }
    return Result;
}

FString UNPCSageBridge::BuildFailurePayload(
    const FString& ErrorCode,
    const FString& StdErr,
    int32 ReturnCode,
    const FString& StdOut
)
{
    return MakeFailurePayload(ErrorCode, StdErr, ReturnCode, StdOut);
}

void UNPCSageBridge::ShutdownBridgeDaemon()
{
    FScopeLock Lock(&GDaemonMutex);

    if (!GDaemonProc.IsValid())
    {
        CleanupStaleIpcFiles();
        return;
    }

    if (!GDaemonShutdownFilePath.IsEmpty())
    {
        WriteFileAtomically(GDaemonShutdownFilePath, TEXT("shutdown"));
    }

    constexpr double GracePeriodSeconds = 1.0;
    const double Start = FPlatformTime::Seconds();
    while (FPlatformProcess::IsProcRunning(GDaemonProc) && (FPlatformTime::Seconds() - Start) < GracePeriodSeconds)
    {
        FPlatformProcess::Sleep(0.01f);
    }

    if (FPlatformProcess::IsProcRunning(GDaemonProc))
    {
        FPlatformProcess::TerminateProc(GDaemonProc, true);
    }

    FPlatformProcess::CloseProc(GDaemonProc);
    GDaemonProc.Reset();

    if (!GDaemonShutdownFilePath.IsEmpty())
    {
        DeleteIfExists(GDaemonShutdownFilePath);
    }

    CleanupStaleIpcFiles();
}

bool UNPCSageBridge::EnsureDaemonRunning(
    const FString& PythonExe,
    const FString& HandlerScript
)
{
    FScopeLock Lock(&GDaemonMutex);

    if (GDaemonProc.IsValid() && FPlatformProcess::IsProcRunning(GDaemonProc))
    {
        return true;
    }

    if (GDaemonProc.IsValid())
    {
        FPlatformProcess::CloseProc(GDaemonProc);
        GDaemonProc.Reset();
    }

    CleanupStaleIpcFiles();

    const FString HandlerPath = ResolveHandlerScriptPath(HandlerScript);
    const FString DaemonPath = FPaths::Combine(FPaths::GetPath(HandlerPath), GDaemonScriptFile);
    if (!FPaths::FileExists(DaemonPath))
    {
        return false;
    }

    const FString ReqDir = BridgeRequestDir();
    const FString RespDir = BridgeResponseDir();
    const FString RuntimeDir = BridgeRuntimeDir();
    GDaemonShutdownFilePath = FPaths::Combine(RuntimeDir, TEXT("daemon.shutdown"));
    DeleteIfExists(GDaemonShutdownFilePath);

    const FString Params = FString::Printf(
        TEXT("%s --request-dir %s --response-dir %s --shutdown-file %s --poll-ms %.2f --idle-sleep-ms %.2f --stale-ms %.2f"),
        *QuoteArg(DaemonPath),
        *QuoteArg(ReqDir),
        *QuoteArg(RespDir),
        *QuoteArg(GDaemonShutdownFilePath),
        BridgePollSeconds() * 1000.0,
        BridgeIdleSleepSeconds() * 1000.0,
        BridgeStaleAgeSeconds() * 1000.0
    );

    uint32 ProcId = 0;
    GDaemonProc = FPlatformProcess::CreateProc(
        *PythonExe,
        *Params,
        true,
        true,
        true,
        &ProcId,
        0,
        nullptr,
        nullptr,
        nullptr,
        nullptr
    );

    if (!GDaemonProc.IsValid())
    {
        return false;
    }

    FPlatformProcess::Sleep(0.05f);
    return FPlatformProcess::IsProcRunning(GDaemonProc);
}

bool UNPCSageBridge::ExecuteHandlerViaDaemon(
    const TArray<FString>& Argv,
    FString& OutStdOut,
    FString& OutStdErr,
    int32& OutReturnCode,
    const FString& PythonExe,
    const FString& HandlerScript
)
{
    OutStdOut.Empty();
    OutStdErr.Empty();
    OutReturnCode = -1;

    if (!EnsureDaemonRunning(PythonExe, HandlerScript))
    {
        OutStdErr = TEXT("daemon_start_failed");
        return false;
    }

    CleanupStaleIpcFiles();

    const FString RequestId = FGuid::NewGuid().ToString(EGuidFormats::Digits);
    const FString RequestPath = FPaths::Combine(BridgeRequestDir(), FString::Printf(TEXT("req_%s.json"), *RequestId));
    const FString ResponsePath = FPaths::Combine(BridgeResponseDir(), FString::Printf(TEXT("resp_%s.json"), *RequestId));

    const TSharedRef<FJsonObject> Req = MakeShared<FJsonObject>();
    Req->SetStringField(TEXT("id"), RequestId);
    Req->SetStringField(TEXT("created_utc"), FDateTime::UtcNow().ToIso8601());

    TArray<TSharedPtr<FJsonValue>> ArgvValues;
    ArgvValues.Reserve(Argv.Num());
    for (const FString& Arg : Argv)
    {
        ArgvValues.Add(MakeShared<FJsonValueString>(Arg));
    }
    Req->SetArrayField(TEXT("argv"), ArgvValues);

    const FString ReqJson = SerializeJsonObject(Req);
    if (!WriteFileAtomically(RequestPath, ReqJson))
    {
        OutStdErr = TEXT("write_request_failed");
        return false;
    }

    const double TimeoutSeconds = BridgeTimeoutSeconds();
    const double PollSeconds = BridgePollSeconds();
    const double Start = FPlatformTime::Seconds();
    while (!FPaths::FileExists(ResponsePath))
    {
        if ((FPlatformTime::Seconds() - Start) >= TimeoutSeconds)
        {
            DeleteIfExists(RequestPath);
            OutStdErr = FString::Printf(TEXT("daemon_timeout_after_%.2fs"), TimeoutSeconds);
            OutReturnCode = -1;
            return false;
        }
        FPlatformProcess::Sleep(static_cast<float>(PollSeconds));
    }

    FString ResponseJson;
    const bool bReadOk = FFileHelper::LoadFileToString(ResponseJson, *ResponsePath);
    DeleteIfExists(ResponsePath);
    DeleteIfExists(RequestPath);

    if (!bReadOk)
    {
        OutStdErr = TEXT("read_response_failed");
        return false;
    }

    if (!ParseDaemonResponse(ResponseJson, OutStdOut, OutStdErr, OutReturnCode))
    {
        OutStdErr = TEXT("parse_response_failed");
        OutStdOut = ResponseJson.Left(512);
        return false;
    }

    return true;
}

bool UNPCSageBridge::ExecuteHandler(
    const TArray<FString>& Argv,
    FString& OutStdOut,
    FString& OutStdErr,
    int32& OutReturnCode,
    const FString& PythonExe,
    const FString& HandlerScript
)
{
    const int32 MaxRetries = BridgeMaxRetries();
    const double BackoffBase = BridgeRetryBackoffSeconds();

    for (int32 Attempt = 0; Attempt <= MaxRetries; ++Attempt)
    {
        FString DaemonStdOut;
        FString DaemonStdErr;
        int32 DaemonReturnCode = -1;

        if (ExecuteHandlerViaDaemon(Argv, DaemonStdOut, DaemonStdErr, DaemonReturnCode, PythonExe, HandlerScript))
        {
            OutStdOut = MoveTemp(DaemonStdOut);
            OutStdErr = MoveTemp(DaemonStdErr);
            OutReturnCode = DaemonReturnCode;
            return true;
        }

        OutStdOut = MoveTemp(DaemonStdOut);
        OutStdErr = MoveTemp(DaemonStdErr);
        OutReturnCode = DaemonReturnCode;

        if (Attempt < MaxRetries)
        {
            ShutdownBridgeDaemon();
            const double Delay = BackoffBase * FMath::Pow(2.0, Attempt);
            FPlatformProcess::Sleep(static_cast<float>(Delay));
        }
    }

    const FString ScriptPath = ResolveHandlerScriptPath(HandlerScript);
    FString Params = QuoteArg(ScriptPath);
    const FString ArgsAsCmd = BuildCommandLineArgs(Argv);
    if (!ArgsAsCmd.IsEmpty())
    {
        Params += TEXT(" ") + ArgsAsCmd;
    }

    return FPlatformProcess::ExecProcess(*PythonExe, *Params, &OutReturnCode, &OutStdOut, &OutStdErr);
}

bool UNPCSageBridge::InvalidatePrefixCache(
    const FString& NPCId,
    const FString& GameStateJson,
    FString& OutResultJson,
    const FString& PythonExe,
    const FString& HandlerScript
)
{
    const TArray<FString> Argv = {
        TEXT("invalidate-prefix-cache"),
        TEXT("--npc-id-b64"), ToBase64Utf8(NPCId),
        TEXT("--game-state-json-b64"), ToBase64Utf8(GameStateJson),
    };

    FString StdOut;
    FString StdErr;
    int32 ReturnCode = -1;
    const bool bExecOk = ExecuteHandler(Argv, StdOut, StdErr, ReturnCode, PythonExe, HandlerScript);
    return FinalizeCommandResult(bExecOk, StdOut, StdErr, ReturnCode, OutResultJson);
}

bool UNPCSageBridge::LoadEpisodicMemory(
    const FString& NPCId,
    const FString& PlayerId,
    const FString& BehaviorState,
    const FString& Query,
    int32 TopK,
    FString& OutResultJson,
    const FString& PythonExe,
    const FString& HandlerScript
)
{
    const int32 SafeTopK = FMath::Clamp(TopK, 1, 12);
    const TArray<FString> Argv = {
        TEXT("load-episodic"),
        TEXT("--npc-id-b64"), ToBase64Utf8(NPCId),
        TEXT("--player-id-b64"), ToBase64Utf8(PlayerId),
        TEXT("--behavior-state-b64"), ToBase64Utf8(BehaviorState),
        TEXT("--query-b64"), ToBase64Utf8(Query),
        TEXT("--top-k"), FString::FromInt(SafeTopK),
    };

    FString StdOut;
    FString StdErr;
    int32 ReturnCode = -1;
    const bool bExecOk = ExecuteHandler(Argv, StdOut, StdErr, ReturnCode, PythonExe, HandlerScript);
    return FinalizeCommandResult(bExecOk, StdOut, StdErr, ReturnCode, OutResultJson);
}

bool UNPCSageBridge::StoreEpisodicMemory(
    const FString& NPCId,
    const FString& Persona,
    const FString& BehaviorState,
    const FString& Location,
    const FString& PlayerInput,
    const FString& NPCResponse,
    const FString& SessionId,
    FString& OutResultJson,
    const FString& PythonExe,
    const FString& HandlerScript
)
{
    const TArray<FString> Argv = {
        TEXT("extract-episodic"),
        TEXT("--npc-id-b64"), ToBase64Utf8(NPCId),
        TEXT("--persona-b64"), ToBase64Utf8(Persona),
        TEXT("--behavior-state-b64"), ToBase64Utf8(BehaviorState),
        TEXT("--location-b64"), ToBase64Utf8(Location),
        TEXT("--player-input-b64"), ToBase64Utf8(PlayerInput),
        TEXT("--npc-response-b64"), ToBase64Utf8(NPCResponse),
        TEXT("--session-id-b64"), ToBase64Utf8(SessionId),
    };

    FString StdOut;
    FString StdErr;
    int32 ReturnCode = -1;
    const bool bExecOk = ExecuteHandler(Argv, StdOut, StdErr, ReturnCode, PythonExe, HandlerScript);
    return FinalizeCommandResult(bExecOk, StdOut, StdErr, ReturnCode, OutResultJson);
}

bool UNPCSageBridge::LoadWorldFacts(
    const FString& NPCId,
    const FString& Location,
    const FString& ActiveQuestPhase,
    int32 MaxFacts,
    FString& OutResultJson,
    const FString& PythonExe,
    const FString& HandlerScript
)
{
    const int32 SafeMaxFacts = FMath::Clamp(MaxFacts, 1, 50);
    const TArray<FString> Argv = {
        TEXT("load-world-facts"),
        TEXT("--npc-id-b64"), ToBase64Utf8(NPCId),
        TEXT("--location-b64"), ToBase64Utf8(Location),
        TEXT("--active-quest-phase-b64"), ToBase64Utf8(ActiveQuestPhase),
        TEXT("--max-facts"), FString::FromInt(SafeMaxFacts),
    };

    FString StdOut;
    FString StdErr;
    int32 ReturnCode = -1;
    const bool bExecOk = ExecuteHandler(Argv, StdOut, StdErr, ReturnCode, PythonExe, HandlerScript);
    return FinalizeCommandResult(bExecOk, StdOut, StdErr, ReturnCode, OutResultJson);
}

bool UNPCSageBridge::PreFetchContext(
    const FString& NPCId,
    const FString& CurrentBehaviorState,
    const FString& Location,
    const FString& GameStateJson,
    int32 TopPredictedStates,
    FString& OutResultJson,
    const FString& PythonExe,
    const FString& HandlerScript
)
{
    const int32 SafeTopPredictedStates = FMath::Clamp(TopPredictedStates, 1, 5);
    const TArray<FString> Argv = {
        TEXT("prefetch-context"),
        TEXT("--npc-id-b64"), ToBase64Utf8(NPCId),
        TEXT("--behavior-state-b64"), ToBase64Utf8(CurrentBehaviorState),
        TEXT("--location-b64"), ToBase64Utf8(Location),
        TEXT("--game-state-json-b64"), ToBase64Utf8(GameStateJson),
        TEXT("--top-predicted-states"), FString::FromInt(SafeTopPredictedStates),
    };

    FString StdOut;
    FString StdErr;
    int32 ReturnCode = -1;
    const bool bExecOk = ExecuteHandler(Argv, StdOut, StdErr, ReturnCode, PythonExe, HandlerScript);
    return FinalizeCommandResult(bExecOk, StdOut, StdErr, ReturnCode, OutResultJson);
}

bool UNPCSageBridge::LoadTrustScore(
    const FString& NPCId,
    const FString& PlayerId,
    FString& OutResultJson,
    const FString& PythonExe,
    const FString& HandlerScript
)
{
    const TArray<FString> Argv = {
        TEXT("load-trust-score"),
        TEXT("--npc-id-b64"), ToBase64Utf8(NPCId),
        TEXT("--player-id-b64"), ToBase64Utf8(PlayerId),
    };

    FString StdOut;
    FString StdErr;
    int32 ReturnCode = -1;
    const bool bExecOk = ExecuteHandler(Argv, StdOut, StdErr, ReturnCode, PythonExe, HandlerScript);
    return FinalizeCommandResult(bExecOk, StdOut, StdErr, ReturnCode, OutResultJson);
}

bool UNPCSageBridge::StoreTrustScore(
    const FString& NPCId,
    const FString& PlayerId,
    float TrustScore,
    const FString& SessionId,
    FString& OutResultJson,
    const FString& PythonExe,
    const FString& HandlerScript
)
{
    const float ClampedTrustScore = FMath::Clamp(TrustScore, -1.0f, 1.0f);
    const TArray<FString> Argv = {
        TEXT("store-trust-score"),
        TEXT("--npc-id-b64"), ToBase64Utf8(NPCId),
        TEXT("--player-id-b64"), ToBase64Utf8(PlayerId),
        TEXT("--session-id-b64"), ToBase64Utf8(SessionId),
        TEXT("--trust-score"), FString::SanitizeFloat(ClampedTrustScore),
    };

    FString StdOut;
    FString StdErr;
    int32 ReturnCode = -1;
    const bool bExecOk = ExecuteHandler(Argv, StdOut, StdErr, ReturnCode, PythonExe, HandlerScript);
    return FinalizeCommandResult(bExecOk, StdOut, StdErr, ReturnCode, OutResultJson);
}

bool UNPCSageBridge::LogImplicitFeedback(
    const FString& NPCId,
    const FString& PlayerId,
    const FString& SessionId,
    float Score,
    const FString& Outcome,
    FString& OutResultJson,
    const FString& PythonExe,
    const FString& HandlerScript
)
{
    const float ClampedScore = FMath::Clamp(Score, -1.0f, 1.0f);
    const TArray<FString> Argv = {
        TEXT("log-feedback"),
        TEXT("--npc-id-b64"), ToBase64Utf8(NPCId),
        TEXT("--player-id-b64"), ToBase64Utf8(PlayerId),
        TEXT("--session-id-b64"), ToBase64Utf8(SessionId),
        TEXT("--score"), FString::SanitizeFloat(ClampedScore),
        TEXT("--outcome-b64"), ToBase64Utf8(Outcome),
    };

    FString StdOut;
    FString StdErr;
    int32 ReturnCode = -1;
    const bool bExecOk = ExecuteHandler(Argv, StdOut, StdErr, ReturnCode, PythonExe, HandlerScript);
    return FinalizeCommandResult(bExecOk, StdOut, StdErr, ReturnCode, OutResultJson);
}

bool UNPCSageBridge::TryGetJsonStringField(
    const FString& JsonPayload,
    const FString& FieldName,
    FString& OutValue
)
{
    OutValue.Empty();
    if (JsonPayload.IsEmpty() || FieldName.IsEmpty())
    {
        return false;
    }

    TSharedPtr<FJsonObject> Root;
    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonPayload);
    if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
    {
        return false;
    }

    if (!Root->HasField(FieldName))
    {
        return false;
    }

    const TSharedPtr<FJsonValue> Value = Root->TryGetField(FieldName);
    if (!Value.IsValid())
    {
        return false;
    }

    if (Value->Type == EJson::String)
    {
        OutValue = Value->AsString();
        return true;
    }
    if (Value->Type == EJson::Boolean)
    {
        OutValue = Value->AsBool() ? TEXT("true") : TEXT("false");
        return true;
    }
    if (Value->Type == EJson::Number)
    {
        OutValue = FString::SanitizeFloat(Value->AsNumber());
        return true;
    }
    if (Value->Type == EJson::Array)
    {
        TArray<TSharedPtr<FJsonValue>> Arr = Value->AsArray();
        OutValue = FString::Printf(TEXT("[array:%d]"), Arr.Num());
        return true;
    }
    if (Value->Type == EJson::Object)
    {
        OutValue = TEXT("{object}");
        return true;
    }
    return false;
}
