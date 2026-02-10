#pragma once

#include "CoreMinimal.h"
#include "Modules/ModuleManager.h"

class FNPCInferenceModule : public IModuleInterface
{
public:
	/** IModuleInterface implementation */
	virtual void StartupModule() override;
	virtual void ShutdownModule() override;

	static inline FNPCInferenceModule& Get()
	{
		return FModuleManager::LoadModuleChecked<FNPCInferenceModule>("NPCInference");
	}

	static inline bool IsAvailable()
	{
		return FModuleManager::Get().IsModuleLoaded("NPCInference");
	}
};
