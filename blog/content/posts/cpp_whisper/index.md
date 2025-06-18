---
title: <C++> Implementing Unreal Plugin for whisper Model.
date: 2025-04-14T18:33:20Z
lastmod: 2025-04-14T18:33:20Z
author: Jun Yeop(Johnny) Na
avatar: /images/favicon.svg
# authorlink: https://author.site
cover: cpp.png
categories:
  - networking
  - ML
tags:
  - cpp
  - voice recognition
# nolastmod: true
draft: false
---

I'm going to make a game that interacts with NPCs with voice.

To do that I need to install a voice recognition AI model which translates user's voice into natural language.

OpenAI has a model called `whisper` which does exactly this, and it has a community C++ extension(https://github.com/ggml-org/whisper.cpp)

I'm setup this whisper.cpp in my game file so that we can run it locally.

# 1. Clone Repository and Download Model Weights

```sh
$ git clone https://github.com/ggml-org/whisper.cpp.git
$ cd whisper.cpp

$ sh ./models/download-ggml-model.cmd base
```

# 2. Build whisper.cpp to Create Static Library Files

## Build whisper.cpp library

```sh
$ mkdir build
$ cd build

$ cmake -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF ..
$ cmake --build . --config Release
```

.lib file is created in `build/src`

![whisper](./whisper.png)

# 3. Create WhisperIntegration Library to use in Unreal Engine 5.5

1. Move model(`ggml-large-v3.bin`) to `Content/Models`
2. Create `Plugins` directory in the base project directory, and create the following structure inside `Plugins`:

```
─WhisperIntegration
    ├─WhisperIntegration.uplugin
    │
    ├─Source
    │  └─WhisperIntegration
    │      ├─  WhisperIntegration.Build.cs
    │      │
    │      ├─Private
    │      │     ├─ WhisperBPLibrary.cpp
    │      │     ├─ WhisperIntegration.cpp
    │      │
    │      └─Public
    │            ├─ WhisperBPLibrary.h
    │            ├─ WhisperIntegration.h
    │
    └─ThirdParty
        └─whisper.cpp
            ├─include
            │    ├─ whisper.h
            │    ├─ ggml.h
			|    ├─ ggml-cpu.h
			|    ├─ ggml-gpu.h
			|    ├─ ggml-backend.h
			|    ├─ ggml-alloc.h
			|    ├─ ggml-cuda.h
			|
            └─lib
                └─Win64
                    ├─ whisper.lib
                    ├─ ggml.lib
                    ├─ ggml-cpu.lib
                    ├─ ggml-cuda.lib
                    ├─ ggml-base.lib

```

1. `whisper.h` will be in `whisper.cpp/include`
2. Other include files are in `whisper.cpp/ggml/include`
3. `whisper.lib` will be in `whisper.cpp/build/src/Release`, ggml libs will be in `whisper.cpp/build/ggml/src/Release`
4. Leave `WhisperComponent.cpp/h` empty for now
5. `WhisperIntegration.Build.cs` is as follows:

```cs
using System.IO;
using UnrealBuildTool;

public class WhisperIntegration : ModuleRules
{
	public WhisperIntegration(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;
		PublicIncludePaths.Add(Path.GetFullPath(Path.Combine(ModuleDirectory, "Public")));
		PrivateIncludePaths.Add(Path.GetFullPath(Path.Combine(ModuleDirectory, "Private")));


		PublicDependencyModuleNames.AddRange(new[] {
			"Core", "CoreUObject", "Engine", "InputCore", "AudioCaptureCore", "SignalProcessing"
		});

		string ThirdPartyPath = Path.Combine(ModuleDirectory, "..", "ThirdParty", "whisper.cpp");
		PublicIncludePaths.Add(Path.Combine(ThirdPartyPath, "include"));

		string LibPath = Path.Combine(ThirdPartyPath, "lib");
		string[] NeededLibs = { "whisper.lib", "ggml.lib", "ggml-base.lib", "ggml-cuda.lib", "ggml-cpu.lib" };

		foreach (string Lib in NeededLibs)
		{
			PublicAdditionalLibraries.Add(Path.Combine(LibPath, Lib));
		}

        string CudaPath = System.Environment.GetEnvironmentVariable("CUDA_PATH");
        if (!string.IsNullOrEmpty(CudaPath))
        {
            string CudaLibPath = Path.Combine(CudaPath, "lib", "x64");
            if (Directory.Exists(CudaLibPath))
            {
                PublicAdditionalLibraries.Add(Path.Combine(CudaLibPath, "cudart.lib"));
                PublicAdditionalLibraries.Add(Path.Combine(CudaLibPath, "cublas.lib"));
                PublicAdditionalLibraries.Add(Path.Combine(CudaLibPath, "cuda.lib"));
                System.Console.WriteLine($"Info: Added CUDA runtime libraries from: {CudaLibPath}");
            }
            else
            {
                System.Console.WriteLine($"Warning: CUDA lib path not found: {CudaLibPath}");
            }
        }
        else
        {
            System.Console.WriteLine("Warning: CUDA_PATH environment variable not found. CUDA libraries will not be linked, which will cause linker errors if CUDA backend is used.");
        }
    }
}
```

6. `WhisperIntegration.uplugin` is as follows:

```json
{
  "FileVersion": 3,
  "Version": 1,
  "VersionName": "1.0",
  "FriendlyName": "Whisper Integration",
  "Description": "Integration of whisper.cpp for speech recognition in UE5",
  "Category": "Audio",
  "CreatedBy": "Your Name",
  "CreatedByURL": "",
  "DocsURL": "",
  "MarketplaceURL": "",
  "SupportURL": "",
  "CanContainContent": true,
  "IsBetaVersion": false,
  "IsExperimentalVersion": false,
  "Installed": false,
  "Modules": [
    {
      "Name": "WhisperIntegration",
      "Type": "Runtime",
      "LoadingPhase": "Default"
    }
  ]
}
```

Now right click on .uproject file and rebuild

# 4. Add Plugins Source

- We have three components in the library:
  - WhisperAudioResampler - whisper.cpp needs 16khz mono channel audio so we define a custom resampler(we can also use AudioDevice Resampler)
  - WhisperBPLibrary - Blueprint Library that defines API for the plugin
  - WhisperIntegrationModule - Core logic for integrating whisper.cpp with Unreal Engine

## Header Files

- `WhisperAudioResampler.h`

```cpp
/* ---------------------------------------------------
	Resamples Audio to fit whisper.cpp
	model's requirement of 16kHz and mono channel
--------------------------------------------------- */

#include "CoreMinimal.h"

class WhisperAudioResampler {
private:
	WhisperAudioResampler();
	~WhisperAudioResampler() {};
	WhisperAudioResampler(const WhisperAudioResampler& resampler) = delete;
	WhisperAudioResampler& operator=(const WhisperAudioResampler& resampler) = delete;
	WhisperAudioResampler& operator=(WhisperAudioResampler&& resampler) = delete;

public:
	static bool Resample(const float* InAudio, int32 NumFrames, int32 NumChannels, int32 StartSampleRate, Audio::FAlignedFloatBuffer& OutAudio);
	static const double WHISPER_TARGET_RESAMPLE_RATE;
};
```

- `WhisperBPLibrary.h`

```cpp
#pragma once

#include "CoreMinimal.h"
#include "Kismet/BlueprintFunctionLibrary.h"
#include "AudioMixerDevice.h"
#include "AudioDeviceManager.h"
#include "Engine/World.h"
#include "WhisperBPLibrary.generated.h"

static TUniquePtr<Audio::FAudioRecordingData> RecordData;

UCLASS()
class WHISPERINTEGRATION_API UWhisperBPLibrary : public UBlueprintFunctionLibrary
{
	GENERATED_BODY()

public:
	UFUNCTION(BlueprintCallable, Category = "Whisper Integration", meta = (WorldContext = "WorldContextObject", DisplayName = "Translate Mic To Text(Whisper)"))
	static FString TranslateMicToText(const UObject* WorldContextObject, USoundSubmix* SubmixToRecord = nullptr);

	UFUNCTION(BlueprintCallable, Category = "Whisper Integration", meta = (DisplayName = "Transcribe Audio Buffer (Whisper)"))
	static FString TranscribeAudioBuffer(const TArray<float> &PCMData);

};
```

- `WhisperIntegrationModule.h`

```cpp
#pragma once

#include "whisper.h"
#include "CoreMinimal.h"
#include "Modules/ModuleInterface.h"

struct whisper_context;

class FWhisperIntegrationModule : public IModuleInterface
{
public:
    virtual void StartupModule() override;
    virtual void ShutdownModule() override;

    static inline FWhisperIntegrationModule& Get() {
        return FModuleManager::GetModuleChecked<FWhisperIntegrationModule>("WhisperIntegration");
    }

    bool InitializeModel(const FString& ModelPath);
    FString TranscribeFromBuffer(const float* PCMData, int32 SampleCount);

private:
    whisper_context* _context = nullptr;
};
```

## Source Files

- `WhisperAudioResampler.cpp`

```cpp
#include "WhisperAudioResampler.h"

const double WhisperAudioResampler::WHISPER_TARGET_RESAMPLE_RATE = 16000.0F;

bool WhisperAudioResampler::Resample(const float* InAudio, int32 NumSamples, int32 NumChannels, int32 StartSampleRate, Audio::FAlignedFloatBuffer& OutAudio)
{
	if (0 == NumSamples) {
		UE_LOG(LogTemp, Error, TEXT("NumSamples == 0. Cannot resample."));
		return false;
	}

	// Interpolate to Mono-Channel
	Audio::FAlignedFloatBuffer MonoChannelAudio = Audio::FAlignedFloatBuffer();
	MonoChannelAudio.SetNum((NumSamples / NumChannels));

	for (int32 sampleCnt = 0; sampleCnt < NumSamples; sampleCnt += NumChannels)
	{
		float sum = 0.f;
		for (int32 channelCnt = 0; channelCnt < NumChannels; ++channelCnt)
			sum += InAudio[sampleCnt + channelCnt];
		MonoChannelAudio.Add(sum / NumChannels);
	}

	// Interpolate to Target Sample Rate
	const double Ratio = double(StartSampleRate) / WHISPER_TARGET_RESAMPLE_RATE;
	const int32 NewLen = FMath::CeilToInt(MonoChannelAudio.Num() / Ratio);

	OutAudio.SetNum(NewLen);

	for (int32 i = 0; i < NewLen; ++i)
	{
		double SrcIdx = i * Ratio;
		int32 Idx = FMath::FloorToInt(SrcIdx);
		double Frac = SrcIdx - Idx;

		float SampleValueAtStart = MonoChannelAudio[FMath::Clamp(Idx, 0, MonoChannelAudio.Num() - 1)];
		float SampleValueAtEnd = MonoChannelAudio[FMath::Clamp(Idx + 1, 0, MonoChannelAudio.Num() - 1)];
		OutAudio[i] = (SampleValueAtStart + (SampleValueAtEnd - SampleValueAtStart) * Frac);
	}

	return true;
}
```

- `WhisperBPLibrary.cpp`

```cpp
#include "WhisperBPLibrary.h"
#include "WhisperIntegrationModule.h"
#include "WhisperAudioResampler.h"

FString UWhisperBPLibrary::TranscribeAudioBuffer(const TArray<float> &PCMData)
{
	if (FModuleManager::Get().IsModuleLoaded(TEXT("WhisperIntegration")))
	{
		return FWhisperIntegrationModule::Get().TranscribeFromBuffer(PCMData.GetData(), PCMData.Num());
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("UWhisperBPLibrary::TranscribeAudioBuffer - WhisperIntegration module is not loaded!"));
		return FString(TEXT("[Error: WhisperIntegration Module Not Loaded]"));
	}
}


FString UWhisperBPLibrary::TranslateMicToText(const UObject* WorldContextObject, USoundSubmix* SubmixToRecord) {
	if (Audio::FMixerDevice* MixerDevice = FAudioDeviceManager::GetAudioMixerDeviceFromWorldContext(WorldContextObject))
	{
		float SampleRate;
		float ChannelCount;

		// Resample to 16kHz, mono channel
		const Audio::FAlignedFloatBuffer& RecordedBuffer = MixerDevice->StopRecording(SubmixToRecord, ChannelCount, SampleRate);
		Audio::FAlignedFloatBuffer ResampledBuffer = Audio::AlignedFloatBuffer();

		const bool resample_result = WhisperAudioResampler::Resample(RecordedBuffer.GetData(), RecordedBuffer.Num(), (int32)ChannelCount, (int32)SampleRate, ResampledBuffer);

		if (!resample_result || 0 == ResampledBuffer.Num())
		{
			UE_LOG(LogTemp, Warning, TEXT("No audio data. Did you call Start Recording Output?"));
			return FString("");
		}


		RecordData.Reset(new Audio::FAudioRecordingData());
		RecordData->InputBuffer = Audio::TSampleBuffer<int16>(ResampledBuffer, 1, 16000);

#if !UE_BUILD_SHIPPING
		RecordData->Writer.BeginWriteToWavFile(RecordData->InputBuffer, "record", "", [SubmixToRecord]()
			{
				if (SubmixToRecord && SubmixToRecord->OnSubmixRecordedFileDone.IsBound())
				{
					SubmixToRecord->OnSubmixRecordedFileDone.Broadcast(nullptr);
				}

				RecordData.Reset();
			}

		);
#endif

		FString output = FWhisperIntegrationModule::Get().TranscribeFromBuffer(ResampledBuffer.GetData(), ResampledBuffer.Num());

		return output;
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("Output recording is an audio mixer only feature."));
	}

	return FString("");

}
```

- `WhisperIntegrationModule.cpp`

```cpp

#include "WhisperIntegrationModule.h"
#include "Misc/Paths.h"
#include "HAL/PlatformMisc.h"

#define LOCTEXT_NAMESPACE "FWhisperIntegrationModule"

void FWhisperIntegrationModule::StartupModule()
{
    UE_LOG(LogTemp, Display, TEXT("WhisperIntegration module starting up"));
    InitializeModel("Models/ggml-base.bin");
}

void FWhisperIntegrationModule::ShutdownModule()
{
    if (_context) {
        whisper_free(_context);
        _context = nullptr;
    }

    UE_LOG(LogTemp, Display, TEXT("WhisperIntegration module shutting down"));
}

bool FWhisperIntegrationModule::InitializeModel(const FString& ModelPath) {
    if (_context) return true;

    FString FullPath = FPaths::ProjectContentDir() / ModelPath;


	whisper_context_params PARAMS;
	PARAMS.use_gpu = true;
	PARAMS.gpu_device = 0;

    _context = whisper_init_from_file_with_params(TCHAR_TO_UTF8(*FullPath), PARAMS);

    if (!_context) {
        UE_LOG(LogTemp, Error, TEXT("Failed to load model in path: %s"), *FullPath);
	}
	else {
		UE_LOG(LogTemp, Display, TEXT("Loaded Model!!!!!!"));
	}

    return _context != nullptr;
}

FString FWhisperIntegrationModule::TranscribeFromBuffer(const float* PCMData, int32 SampleCount)
{
	if (!_context)
	{
		UE_LOG(LogTemp, Error, TEXT("Whisper model not initialized!"));
		return FString();
	}


	whisper_full_params Params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
	Params.n_threads = FMath::Max(1, FPlatformMisc::NumberOfCoresIncludingHyperthreads());
	Params.offset_ms = 0;
	Params.translate = false;
	Params.print_realtime = false;
	Params.logprob_thold = -1.0f;
	Params.no_speech_thold = 0.0f;
	Params.language = "ko";

	int32 Success = whisper_full(_context, Params, PCMData, SampleCount);
	if (Success != 0)
	{
		UE_LOG(LogTemp, Error, TEXT("whisper_full failed: %d"), Success);
		return FString();
	}

    FString ResultText = TEXT("");
    const int NumSegments = whisper_full_n_segments(_context);

    for (int32 i = 0; i < NumSegments; ++i) {
        const char* SegmentTextAnsi = whisper_full_get_segment_text(_context, i);
        if (SegmentTextAnsi)
        {
            ResultText += FString(UTF8_TO_TCHAR(SegmentTextAnsi));
        }
    }

	return ResultText;
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FWhisperIntegrationModule, WhisperIntegration)
```

# 5. Add Recording and Whisper API to Character

1. We create a submix and name it `Record`

![submix](./createsubmix.png)

2. We create a `AudioCaptureComponent` in Character and Connect the following blueprints:

![acc](./audiocapture.png)

Now when we start the game and press 1 -> speak -> release 1, the `whisper` model transcribes what we said into natural language.
