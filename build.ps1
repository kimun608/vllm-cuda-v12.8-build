[CmdletBinding()]
param(
    [string]$Version,
    [string]$Image,
    [string]$CudaArchitectures,
    [ValidateRange(1, 1024)][int]$MaxJobs,
    [ValidateRange(1, 1024)][int]$TorchMaxJobs,
    [ValidateRange(1, 1024)][int]$VllmMaxJobs,
    [ValidateRange(1, 1024)][int]$NvccThreads,
    [switch]$PreflightOnly,
    [switch]$SkipPreflight,
    [switch]$PrintConfig,
    [Parameter(ValueFromRemainingArguments = $true)][string[]]$DockerBuildArgs = @()
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
foreach ($argument in $DockerBuildArgs) {
    if ($argument -eq '--build-arg' -or $argument.StartsWith('--build-arg=')) {
        throw '--build-arg overrides are not accepted here. Use versions.env, .env, shell variables, or the dedicated PowerShell parameters so preflight and compilation use the same settings.'
    }
}
. (Join-Path $PSScriptRoot 'scripts/load-config.ps1')
$overrides = @{}
$parameterKeys = @{
    Version='VLLM_VERSION'; Image='VLLM_IMAGE'; CudaArchitectures='TORCH_CUDA_ARCH_LIST';
    MaxJobs='MAX_JOBS'; TorchMaxJobs='TORCH_MAX_JOBS'; VllmMaxJobs='VLLM_MAX_JOBS'; NvccThreads='NVCC_THREADS'
}
foreach ($parameter in $parameterKeys.Keys) {
    if ($PSBoundParameters.ContainsKey($parameter)) { $overrides[$parameterKeys[$parameter]] = $PSBoundParameters[$parameter] }
}
$config = Get-VllmConfig -Root $PSScriptRoot -Overrides $overrides
if ($PrintConfig) { Write-VllmConfig $config; exit 0 }
if ($PreflightOnly -and $SkipPreflight) { throw '-PreflightOnly and -SkipPreflight cannot be combined' }
Get-Command docker -ErrorAction Stop | Out-Null
$dockerOs = & docker info --format '{{.OSType}}'
if ($LASTEXITCODE -ne 0) { throw 'Cannot connect to Docker. Start the Linux Docker engine.' }
if ($dockerOs.Trim() -ne 'linux') { throw 'Use Linux containers (Windows: Docker Desktop with WSL2).' }
$buildArguments = @()
foreach ($key in (Get-VllmBuildKeys)) { $buildArguments += @('--build-arg', "$key=$($config[$key])") }

function Invoke-LoggedDockerBuild {
    param([string[]]$Arguments, [string]$LogPath)
    $ErrorActionPreference = 'Continue'
    & docker @Arguments 2>&1 | Tee-Object -FilePath $LogPath -ErrorAction Stop
    $result = $LASTEXITCODE
    if ($result -ne 0) {
        Write-Host "Build failed (exit $result). See $LogPath."
        exit $result
    }
}

$previousBuildKit = $env:DOCKER_BUILDKIT
Push-Location -LiteralPath $PSScriptRoot
try {
    New-Item -ItemType Directory -Path 'artifacts' -Force | Out-Null
    $env:DOCKER_BUILDKIT = '1'
    # Fixed arguments come last so extra Docker flags cannot change the checked stack.
    $commonArguments = @('build') + $DockerBuildArgs + @('--platform', 'linux/amd64', '--progress', 'plain') + $buildArguments
    if (!$SkipPreflight) {
        $preflightImage = "local/vllm-cu128-dependency-preflight:$($config['VLLM_VERSION'])"
        Write-Host "Checking dependency compatibility for vLLM $($config['VLLM_VERSION'])"
        Invoke-LoggedDockerBuild -Arguments ($commonArguments + @('--file', 'Dockerfile.preflight', '--tag', $preflightImage, '.')) -LogPath 'artifacts/preflight.log'
    }
    if ($PreflightOnly) { Write-Host 'Dependency preflight passed. No source compilation was started.'; exit 0 }
    Write-Host "Building $($config['VLLM_IMAGE']) for CUDA architectures $($config['TORCH_CUDA_ARCH_LIST'])"
    Invoke-LoggedDockerBuild -Arguments ($commonArguments + @('--file', 'Dockerfile', '--target', 'vllm-openai', '--tag', $config['VLLM_IMAGE'], '.')) -LogPath 'artifacts/build.log'
    Write-Host "Built $($config['VLLM_IMAGE']). Log: artifacts/build.log"
}
finally {
    $env:DOCKER_BUILDKIT = $previousBuildKit
    Pop-Location
}
