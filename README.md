# vLLM CUDA 12.8 소스 빌드

vLLM과 PyTorch를 CUDA 12.8로 빌드하는 Linux amd64 Docker 구성입니다. Windows에서는 Docker Desktop의 WSL2 Linux 컨테이너를 사용합니다. 기본 컴파일 대상은 H200(`sm_90`)과 RTX 5090(`sm_120`)이며, **H200 실물 GPU 검증은 수행하지 않았습니다.**

이 저장소에는 재빌드용 설정·스크립트·검증 코드만 포함합니다. 모델 가중치, 완성 이미지 TAR, 빌드 캐시, 벤치마크 결과는 포함하지 않습니다. 런타임 작업 경로는 `/vllm-workspace`입니다.

## 준비와 복제

Git, Linux 컨테이너를 빌드할 Docker/BuildKit, 인터넷 연결과 충분한 메모리·디스크 공간이 필요합니다. PyTorch부터 소스 컴파일하므로 첫 빌드는 오래 걸릴 수 있습니다. 빌드 자체에는 GPU가 필요하지 않습니다.

GPU 검증에는 NVIDIA 드라이버와 GPU 컨테이너 지원이 필요합니다. Linux는 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), Windows는 [Docker Desktop의 WSL2 GPU 지원](https://docs.docker.com/desktop/features/gpu/)을 구성하세요. 네이티브 Windows 바이너리를 만드는 구성은 아닙니다.

```console
git clone https://github.com/kimun608/vllm-cuda-v12.8-build.git
cd vllm-cuda-v12.8-build
```

## 버전과 설정

[`versions.env`](versions.env)가 공통 설정입니다. 기본 조합은 vLLM 0.32.0, PyTorch 2.13.0, torchvision 0.28.0, torchaudio 2.11.0, Triton 3.7.1, torchcodec 0.14.0, CUDA 12.8.1입니다.

2026-10-19에 이 조합으로 소스 빌드를 완료하고 RTX 5090 Laptop(SM120)·Docker Desktop/WSL2에서 기본 GPU 연산, NVFP4 36개 사례, Qwen3-0.6B API 응답을 검증했습니다. Linux/amd64 이미지 TAR의 36개 레이어도 검사했습니다. H200 및 SM90 전용 커널 실행은 별도 검증이 필요합니다.

- 우선순위는 `versions.env` → 선택적인 `.env` → 셸 환경 변수 → 명시한 PowerShell 인자입니다. 필요하면 `.env.example`을 `.env`로 복사하세요.
- 설정은 `KEY=VALUE` 형식입니다. 셸 실행·변수 치환·줄 끝 주석은 지원하지 않습니다. `TORCH_CUDA_ARCH_LIST=9.0;12.0`은 그대로 읽습니다.
- `VLLM_IMAGE`가 비어 있으면 `<VLLM_IMAGE_REPOSITORY>:v<VLLM_VERSION>-cu128`이 됩니다. 기본 태그는 `lottecard/vllm-openai:v0.32.0-cu128`이며 로컬 빌드 결과의 이름입니다.
- 소스는 `v<VERSION>` 릴리스 태그에서 가져옵니다. 선택적인 `*_COMMIT`에 전체 40자리 SHA를 지정하면 태그와 일치하는지 확인합니다.
- 버전을 바꿀 때 관련 라이브러리와 `VLLM_REQUIRED_EXTENSIONS`도 해당 릴리스에 맞춰야 합니다. 최신 버전을 자동 추적하지 않습니다.

새 버전의 CUDA 12.8 호환성을 보장하지 않습니다. 사전 검사, 실제 컴파일, 대상 GPU 검증을 차례로 통과해야 합니다.

## Linux에서 빌드

```bash
# Docker 없이 설정과 최종 이미지 이름 확인
bash build.sh --print-config

# 의존성 검사만 실행
bash build.sh --preflight-only

# 사전 검사 후 전체 소스 빌드
bash build.sh
```

환경 변수로 일회성 설정을 전달할 수 있습니다. H200 전용이면 아키텍처를 `9.0`으로 줄일 수 있습니다.

```bash
TORCH_CUDA_ARCH_LIST=9.0 TORCH_MAX_JOBS=2 VLLM_MAX_JOBS=1 bash build.sh
VLLM_VERSION=0.32.0 bash build.sh --print-config
```

## Windows에서 빌드

Docker Desktop을 WSL2 Linux 컨테이너 모드로 실행한 뒤 저장소 디렉터리의 PowerShell에서 실행합니다. PowerShell 7에서는 `powershell` 대신 `pwsh`를 사용해도 됩니다.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1 -PrintConfig
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1 -PreflightOnly
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\build.ps1 -Version 0.32.0 -TorchMaxJobs 2 -VllmMaxJobs 1 -CudaArchitectures '9.0;12.0'
```

`-ExecutionPolicy Bypass`는 해당 새 프로세스에만 적용하며 시스템 실행 정책을 바꾸지 않습니다.

## 사전 검사와 빌드 자원

래퍼는 `Dockerfile.preflight`에 본 빌드와 같은 버전 설정을 전달합니다. 태그·의존성 검사가 실패하면 무거운 소스 컴파일을 시작하지 않습니다. 사전 검사 통과만으로 CUDA 컴파일이나 추론 성공이 확인되는 것은 아닙니다.

로그는 실행하면서 `artifacts/preflight.log`, `artifacts/build.log`에 생성합니다. 명시적인 `--skip-preflight` / `-SkipPreflight`로 검사를 생략할 수 있습니다. 래퍼에 `--build-arg`를 직접 전달하는 것은 거부하므로 설정 파일·환경 변수·전용 인자를 사용하세요.

기본 병렬도는 `TORCH_MAX_JOBS=2`, `VLLM_MAX_JOBS=2`, `NVCC_THREADS=1`입니다. 메모리가 부족하면 병렬도를 낮추세요. Windows에서는 WSL2에 할당된 메모리와 디스크 여유도 확인하세요.

컴파일 캐시는 버전·소스 커밋·툴체인·GPU 대상·빌드 스크립트별로 구분합니다. 병렬도 변경만으로는 캐시를 무효화하지 않지만 새 버전이나 구성의 첫 빌드에서 이전 객체의 재사용을 보장하지 않습니다.

선택적으로 전용 BuildKit 빌더를 만들 수 있습니다. 제공된 설정은 캐시 기준 120GiB와 여유 공간 기준 40GiB를 사용하므로 호스트 용량에 맞게 검토하세요. 캐시는 이 빌더의 Docker 볼륨에 저장됩니다.

```console
docker buildx create --name vllm-cu128 --driver docker-container --buildkitd-config scripts/buildkitd.toml --bootstrap
```

```bash
bash build.sh --builder vllm-cu128 --load
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "& '.\build.ps1' -DockerBuildArgs '--builder','vllm-cu128','--load'"
```

## 빌드 후 확인

아래 예시의 이미지 태그는 실제 설정한 태그로 바꾸세요. 이 검사는 모델을 다운로드하지 않습니다.

```console
docker run --rm --entrypoint cat lottecard/vllm-openai:v0.32.0-cu128 /opt/build-info.json
docker run --rm --gpus all --entrypoint /opt/.venv/bin/python lottecard/vllm-openai:v0.32.0-cu128 /opt/verify-stack.py --gpu
```

`build-info.json`에는 실제 패키지 버전·소스 커밋·GPU 대상·호환성 패치가 기록됩니다. GPU 검사는 PyTorch 연산, Triton 컴파일, vLLM CUDA 커널을 실행합니다. `nvidia-smi`의 CUDA Version은 드라이버의 지원 상한이며 실제 빌드 버전은 `torch.version.cuda`와 이미지의 `nvcc`로 확인합니다.

`tests/`는 CUDA 패치와 소스 캐시의 회귀검사입니다. `cooperative-topk-runtime.py`는 실제 설치 버전에 따라 0.31.0의 `csrc/libtorch_stable` 또는 0.32.0의 `csrc`를 선택하고, 이미지에 기록된 소스 커밋·파일 해시·패치 내역을 검토한 값과 대조합니다. 실행 검증 범위는 H200/H100의 SM90이며 다른 GPU는 건너뜁니다. 이는 upstream 커널의 다른 GPU 지원 여부를 뜻하지 않습니다. 다른 릴리스는 테스트의 소스 검증 목록도 검토해야 합니다.

Windows에서 선택적으로 `scripts/finish-build.ps1`을 사용하면 이미지·GPU·API 검증 후 TAR와 SHA256 파일을 생성할 수 있습니다. **이미 다운로드한 모델 캐시와 호스트 Python이 필요합니다.** 이 단계는 오프라인이며 모델을 받지 않습니다.

`VLLM_CACHE_DIR`로 호스트 Hugging Face 캐시를 지정하고, 아래 `YOUR_CACHED_REVISION`을 그 캐시에 실제로 존재하는 snapshot 이름으로 바꾸세요. `-SmokeModelPath`는 컨테이너의 `/root/.cache/huggingface/` 아래 경로입니다.

```powershell
$env:VLLM_CACHE_DIR = (Resolve-Path .\.cache\huggingface).Path
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\finish-build.ps1 -Image lottecard/vllm-openai:v0.32.0-cu128 -PythonExecutable python -SmokeModel Qwen/Qwen3-0.6B -SmokeModelPath /root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/YOUR_CACHED_REVISION
```

선택한 버전에 맞는 `artifacts/runtime-assets-manifest-v<VERSION>.json`이 있거나 `-RuntimeAssetsManifest`로 지정하면, GPU 검사 전에 공식 이미지에서 작성한 파일 해시 목록으로 런타임 파일·템플릿·Rust frontend를 CPU에서 검사합니다. manifest의 `vllm_version`은 선택한 버전과 같아야 하며, 해당 manifest가 없으면 이 선택 검사를 건너뜁니다.

이 선택 단계에서 `artifacts/completion-*`, 상태 JSON, 이미지 TAR를 새로 생성합니다. 해당 파일은 저장소에 포함하지 않습니다. 기본 API 검증은 BF16·FlashAttention 2를 명시합니다. 별도로 띄운 서버는 표준 라이브러리 전용 `scripts/smoke-api.py`로 검사할 수도 있습니다.

## 작업 경로만 변경하는 기존 이미지

전체 `Dockerfile` 빌드는 이미 `WORKDIR /vllm-workspace`를 사용합니다. 기존 로컬 이미지의 작업 경로만 바꾸려면 `Dockerfile.workspace`를 선택적으로 사용합니다. `BASE_IMAGE`는 로컬에 있거나 접근 가능한 레지스트리에 있어야 하며, 아래 기본 태그가 공개 배포되었다는 뜻은 아닙니다.

```console
docker build -f Dockerfile.workspace --build-arg BASE_IMAGE=lottecard/vllm-openai:v0.32.0-cu128 -t lottecard/vllm-openai:v0.32.0-cu128-workspace .
```

이 얇은 이미지는 `WORKDIR`만 바꿉니다. 기존 환경 변수·캐시 경로·사용자·권한 정책·entrypoint를 유지하며 추가 에이전트를 설치하지 않습니다.

추가 serving 패키지 핀은 공식 0.31.0 이미지에서 가져와 0.32.0과의 의존성 호환성을 검토한 조합입니다. 공식 0.32.0과 전체 패키지가 같다는 의미는 아닙니다. 설치 전 CUDA 12.8 환경의 패키지 버전을 제약으로 동결해 유지하고 `uv pip check`로 의존성을 검사하며, 최종 목록은 이미지의 `/opt/installed-packages.txt`에 기록합니다.

## CUDA 12.8 호환 범위

- 시스템 툴킷, PyTorch/vLLM 컴파일과 Triton의 일반·Blackwell용 `ptxas`에 CUDA 12.8을 적용합니다. torchcodec은 CPU wheel을 사용합니다. 선택 의존성 CUTLASS DSL에는 더 높은 버전의 컴파일러·분석 도구가 포함될 수 있어 모든 포함 바이너리가 12.8이라는 뜻은 아닙니다.
- vLLM 0.31.0/0.32.0의 DeepGEMM은 CUDA Runtime 12.9 이상을 요구하므로 빌드와 패키징에서 제외합니다. DeepGEMM을 직접 import하는 `Dots3NoteForCausalLM`, `Dots3NoteMTP`는 지원하지 않습니다.
- FlashInfer 0.7의 Blackwell `compute_120f` JIT 경로에는 CUDA 12.9 이상이 필요합니다. CUDA 12.8에서 해당 경로는 지원 대상으로 삼지 않으며, 제공된 API 검증은 `--attention-backend FLASH_ATTN --attention-config.flash_attn_version=2`를 기본으로 사용합니다.
- 일부 FlashInfer sampler/all-reduce·FP8·DeepGEMM 실행 경로는 환경 변수로 비활성화합니다. FlashMLA·DeepSelect는 해당 릴리스의 패키징 조건에 따르며 DeepEP는 포함하지 않습니다. 모델·양자화·expert parallel 조합은 별도 검증이 필요합니다.
- `prepare-cudnn-frontend.py`는 검토한 공식 wheel의 해시를 검증하고 코드·라이브러리를 유지한 채 CUTLASS의 `cu13` 의존성 메타데이터를 조정합니다. 수정 버전과 해시는 이미지 안에 기록하며 새로운 upstream wheel은 다시 검토해야 합니다.
- `apply-cu128-patches.py`는 Cooperative TopK의 CUDA 12.8 PTX 래퍼, NVFP4 타입 호환성, 생성 소스 재시도 캐시 및 DeepGEMM 조건을 수정합니다. 알려진 패치 지점이 일부만 일치하면 실패하므로 새 릴리스는 소스 검토가 필요합니다.
