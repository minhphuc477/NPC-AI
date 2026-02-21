// NPCInference.Build.cs - Build configuration for NPC Inference plugin

using UnrealBuildTool;
using System.IO;

public class NPCInference : ModuleRules
{
	public NPCInference(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;
		
		// C++ standard
		CppStandard = CppStandardVersion.Cpp17;
		
		PublicIncludePaths.AddRange(
			new string[] {
				// Add public include paths here
			}
		);
				
		
		PrivateIncludePaths.AddRange(
			new string[] {
				// Add private include paths here
			}
		);
			
		
		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"Json",
				"JsonUtilities",
				"HTTP"
			}
		);
			
		
		PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				// Add private dependencies here
			}
		);
		
		
		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
				// Add dynamically loaded modules here
			}
		);

		// Path to NPC AI C++ inference engine
		string NPCAIPath = Path.GetFullPath(Path.Combine(ModuleDirectory, "..", "..", "..", "..", "..", "cpp"));
		string NPCAIIncludePath = Path.Combine(NPCAIPath, "include");
		string NPCAILibPath = Path.Combine(NPCAIPath, "lib");
		
		// Add C++ inference engine headers
		PublicIncludePaths.Add(NPCAIIncludePath);
		
		// Add nlohmann-json (header-only, should be in cpp/include or cpp/lib)
		string JsonIncludePath = Path.Combine(NPCAIPath, "build", "_deps", "json-src", "include");
		if (Directory.Exists(JsonIncludePath))
		{
			PublicIncludePaths.Add(JsonIncludePath);
		}
		
		// Add SentencePiece
		string SentencePieceInclude = Path.Combine(NPCAIPath, "build", "_deps", "sentencepiece-src", "src");
		string SentencePieceBuildInclude = Path.Combine(NPCAIPath, "build", "_deps", "sentencepiece-build", "src");
		
		if (Directory.Exists(SentencePieceInclude))
		{
			PublicIncludePaths.Add(SentencePieceInclude);
		}
		if (Directory.Exists(SentencePieceBuildInclude))
		{
			PublicIncludePaths.Add(SentencePieceBuildInclude);
		}
		
		// Link SentencePiece library
		if (Target.Platform == UnrealTargetPlatform.Win64)
		{
			string SentencePieceLib = Path.Combine(NPCAIPath, "build", "_deps", "sentencepiece-build", "src", "Release", "sentencepiece.lib");
			if (File.Exists(SentencePieceLib))
			{
				PublicAdditionalLibraries.Add(SentencePieceLib);
			}
			
			// Link NPC Inference Lib
			string NPCInferenceLib = Path.Combine(NPCAILibPath, "npc_inference.lib");
			if (File.Exists(NPCInferenceLib))
			{
				PublicAdditionalLibraries.Add(NPCInferenceLib);
			}

            // Link ONNX Runtime Lib
			string OnnxRuntimeLib = Path.Combine(NPCAILibPath, "onnxruntime.lib");
			if (File.Exists(OnnxRuntimeLib))
			{
				PublicAdditionalLibraries.Add(OnnxRuntimeLib);
			}

			// Also need the DLL at runtime
			string SentencePieceDLL = Path.Combine(NPCAIPath, "build", "_deps", "sentencepiece-build", "src", "Release", "sentencepiece.dll");
			if (File.Exists(SentencePieceDLL))
			{
				RuntimeDependencies.Add(SentencePieceDLL);
			}

			// Deploy ONNX Runtime DLL for packaged Game Builds
			string OnnxRuntimeDLL = Path.Combine(NPCAILibPath, "onnxruntime", "lib", "onnxruntime.dll");
			if (!File.Exists(OnnxRuntimeDLL)) 
			{
				// Fallback to build dir
				OnnxRuntimeDLL = Path.Combine(NPCAIPath, "build", "Release", "onnxruntime.dll");
			}
			if (File.Exists(OnnxRuntimeDLL))
			{
				RuntimeDependencies.Add(OnnxRuntimeDLL);
			}
		}
		
		// Disable warnings from third-party headers
		bEnableUndefinedIdentifierWarnings = false;
		bEnableExceptions = true;
	}
}
