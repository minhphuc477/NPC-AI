// NPCSageWorldFactSubsystem.cpp

#include "NPCSageWorldFactSubsystem.h"

#include "Misc/DateTime.h"

namespace
{
FString UtcIsoNow()
{
    return FDateTime::UtcNow().ToIso8601();
}
} // namespace

void UNPCSageWorldFactSubsystem::BroadcastFact(const FString& Key, const FString& Value, FName NPCID)
{
    const FString CanonKey = Key.TrimStartAndEnd();
    const FString CanonValue = Value.TrimStartAndEnd();
    if (CanonKey.IsEmpty() || CanonValue.IsEmpty())
    {
        return;
    }
    FNPCSageWorldFactRecord Row;
    Row.Key = CanonKey;
    Row.Value = CanonValue;
    Row.SourceNPC = NPCID;
    Row.UpdatedUtc = UtcIsoNow();
    FactsByKey.Add(CanonKey, MoveTemp(Row));
}

bool UNPCSageWorldFactSubsystem::GetFact(const FString& Key, FString& OutValue) const
{
    OutValue.Empty();
    const FString CanonKey = Key.TrimStartAndEnd();
    if (CanonKey.IsEmpty())
    {
        return false;
    }
    const FNPCSageWorldFactRecord* Row = FactsByKey.Find(CanonKey);
    if (!Row)
    {
        return false;
    }
    OutValue = Row->Value;
    return true;
}

bool UNPCSageWorldFactSubsystem::CheckConflict(const FString& Key, const FString& ProposedValue) const
{
    FString Existing;
    if (!GetFact(Key, Existing))
    {
        return false;
    }
    const FString Left = Existing.TrimStartAndEnd().ToLower();
    const FString Right = ProposedValue.TrimStartAndEnd().ToLower();
    return !Left.IsEmpty() && !Right.IsEmpty() && Left != Right;
}

TArray<FNPCSageWorldFactRecord> UNPCSageWorldFactSubsystem::GetAllFacts() const
{
    TArray<FNPCSageWorldFactRecord> Out;
    Out.Reserve(FactsByKey.Num());
    for (const TPair<FString, FNPCSageWorldFactRecord>& It : FactsByKey)
    {
        Out.Add(It.Value);
    }
    return Out;
}

FString UNPCSageWorldFactSubsystem::BuildSummaryForNPC(FName NPCID, int32 MaxFacts) const
{
    const int32 SafeMaxFacts = FMath::Clamp(MaxFacts, 1, 50);
    TArray<const FNPCSageWorldFactRecord*> Rows;
    Rows.Reserve(FactsByKey.Num());
    for (const TPair<FString, FNPCSageWorldFactRecord>& It : FactsByKey)
    {
        Rows.Add(&It.Value);
    }
    Rows.Sort([](const FNPCSageWorldFactRecord* A, const FNPCSageWorldFactRecord* B)
    {
        if (!A || !B)
        {
            return A != nullptr;
        }
        return A->UpdatedUtc > B->UpdatedUtc;
    });

    TArray<FString> Parts;
    Parts.Reserve(SafeMaxFacts);
    for (const FNPCSageWorldFactRecord* Row : Rows)
    {
        if (!Row)
        {
            continue;
        }
        if (!NPCID.IsNone() && Row->SourceNPC == NPCID)
        {
            continue;
        }
        Parts.Add(Row->Value);
        if (Parts.Num() >= SafeMaxFacts)
        {
            break;
        }
    }
    return FString::Join(Parts, TEXT("; "));
}
