#include "NPCInferenceModule.h"

#define LOCTEXT_NAMESPACE "FNPCInferenceModule"

void FNPCInferenceModule::StartupModule()
{
	// This code will execute after your module is loaded into memory
}

void FNPCInferenceModule::ShutdownModule()
{
	// This function may be called during shutdown to clean up your module
}

#undef LOCTEXT_NAMESPACE
	
IMPLEMENT_MODULE(FNPCInferenceModule, NPCInference)
