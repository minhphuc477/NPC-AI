// NPCDialogueClient.h - Unreal Engine 5 HTTP Client for BD-NSCA NPC Dialogue
// Add to your UE5 project's Source folder, include "Http" module in Build.cs
#pragma once

#include "CoreMinimal.h"
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"
#include "Json.h"
#include "JsonUtilities.h"

DECLARE_DELEGATE_TwoParams(FOnNPCResponse, bool /* Success */, const FString& /* Response */);

/**
 * Client for communicating with BD-NSCA NPC Dialogue Server
 * Usage:
 *   UNPCDialogueClient* Client = NewObject<UNPCDialogueClient>();
 *   Client->SetServerURL("http://localhost:8080");
 *   Client->RequestDialogue(Context, PlayerInput, FOnNPCResponse::CreateLambda([](bool Success, const FString& Response) {
 *       if (Success) { UE_LOG(LogTemp, Log, TEXT("NPC: %s"), *Response); }
 *   }));
 */
UCLASS()
class YOURGAME_API UNPCDialogueClient : public UObject
{
    GENERATED_BODY()

public:
    /** Server URL (default: http://localhost:8080) */
    UPROPERTY(EditAnywhere, BlueprintReadWrite)
    FString ServerURL = "http://localhost:8080";

    /** Request NPC dialogue response */
    UFUNCTION(BlueprintCallable, Category = "NPC Dialogue")
    void RequestDialogue(
        const FString& NPCID,
        const FString& Persona,
        const FString& Scenario,
        const FString& BehaviorState,
        const FString& PlayerInput,
        FOnNPCResponse OnResponse
    )
    {
        TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
        Request->SetURL(ServerURL + "/generate");
        Request->SetVerb("POST");
        Request->SetHeader("Content-Type", "application/json");

        // Build JSON payload
        TSharedPtr<FJsonObject> JsonObject = MakeShareable(new FJsonObject);
        TSharedPtr<FJsonObject> ContextObj = MakeShareable(new FJsonObject);
        
        ContextObj->SetStringField("npc_id", NPCID);
        ContextObj->SetStringField("persona", Persona);
        ContextObj->SetStringField("scenario", Scenario);
        ContextObj->SetStringField("behavior_state", BehaviorState);
        
        JsonObject->SetObjectField("context", ContextObj);
        JsonObject->SetStringField("player_input", PlayerInput);

        FString OutputString;
        TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&OutputString);
        FJsonSerializer::Serialize(JsonObject.ToSharedRef(), Writer);
        
        Request->SetContentAsString(OutputString);
        
        Request->OnProcessRequestComplete().BindLambda(
            [OnResponse](FHttpRequestPtr Req, FHttpResponsePtr Resp, bool bSuccess)
            {
                if (bSuccess && Resp.IsValid() && Resp->GetResponseCode() == 200)
                {
                    TSharedPtr<FJsonObject> JsonResponse;
                    TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(Resp->GetContentAsString());
                    
                    if (FJsonSerializer::Deserialize(Reader, JsonResponse))
                    {
                        FString NPCResponse = JsonResponse->GetStringField("response");
                        OnResponse.ExecuteIfBound(true, NPCResponse);
                        return;
                    }
                }
                OnResponse.ExecuteIfBound(false, "Failed to get NPC response");
            }
        );
        
        Request->ProcessRequest();
    }

    /** Check server health */
    UFUNCTION(BlueprintCallable, Category = "NPC Dialogue")
    void CheckHealth(FOnNPCResponse OnResponse)
    {
        TSharedRef<IHttpRequest, ESPMode::ThreadSafe> Request = FHttpModule::Get().CreateRequest();
        Request->SetURL(ServerURL + "/health");
        Request->SetVerb("GET");
        
        Request->OnProcessRequestComplete().BindLambda(
            [OnResponse](FHttpRequestPtr Req, FHttpResponsePtr Resp, bool bSuccess)
            {
                if (bSuccess && Resp.IsValid() && Resp->GetResponseCode() == 200)
                {
                    OnResponse.ExecuteIfBound(true, "Server is healthy");
                }
                else
                {
                    OnResponse.ExecuteIfBound(false, "Server not available");
                }
            }
        );
        
        Request->ProcessRequest();
    }
};
