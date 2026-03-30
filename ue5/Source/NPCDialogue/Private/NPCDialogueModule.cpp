#include "Modules/ModuleManager.h"
#include "NPCSageBridge.h"

class FNPCDialogueModule : public IModuleInterface
{
public:
    virtual void StartupModule() override {}
    virtual void ShutdownModule() override
    {
        UNPCSageBridge::ShutdownBridgeDaemon();
    }
};

IMPLEMENT_MODULE(FNPCDialogueModule, NPCDialogue)
