using UnrealBuildTool;

public class NPCDialogue : ModuleRules
{
    public NPCDialogue(ReadOnlyTargetRules Target) : base(Target)
    {
        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PublicDependencyModuleNames.AddRange(
            new[]
            {
                "Core",
                "CoreUObject",
                "Engine",
                "AIModule",
                "GameplayTasks",
                "Json",
                "JsonUtilities",
            }
        );
    }
}
