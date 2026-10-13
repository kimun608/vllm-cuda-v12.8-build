# Validate and export the configured Linux CUDA 12.8 image, optionally waiting
# for an already running Docker build. Run in a durable PowerShell process.
[CmdletBinding()]
param(
    [ValidateRange(0, 2147483647)][int]$BuildProcessId = 0,
    [DateTimeOffset]$BuildStartedAt,
    [string]$Version,
    [string]$Image,
    [string]$RootPath = (Split-Path -Parent $PSScriptRoot),
    [string]$PythonExecutable,
    [ValidateRange(1, 65535)][int]$SmokePort = 18030,
    [string]$SmokeModel = 'Qwen/Qwen3-0.6B',
    [string]$SmokeModelPath = '/root/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca'
)
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $false
$rootPath = (Resolve-Path -LiteralPath $RootPath).Path
. (Join-Path $rootPath 'scripts/load-config.ps1')
$overrides = @{}
if ($PSBoundParameters.ContainsKey('Version')) { $overrides['VLLM_VERSION'] = $Version }
if ($PSBoundParameters.ContainsKey('Image')) { $overrides['VLLM_IMAGE'] = $Image }
$config = Get-VllmConfig -Root $rootPath -Overrides $overrides
$selectedVersion = $config['VLLM_VERSION']
$imageTag = $config['VLLM_IMAGE']
if ($imageTag -match '\s' -or $imageTag.StartsWith('-') -or $imageTag.Contains('@')) {
    throw 'VLLM_IMAGE must be a tagged image reference, without whitespace or a digest'
}
if (!$SmokeModel.Trim() -or !$SmokeModelPath.StartsWith('/root/.cache/huggingface/')) {
    throw 'Supply a nonempty smoke model ID and an absolute path inside the mounted Hugging Face cache'
}
Get-Command docker -ErrorAction Stop | Out-Null
if (!$PythonExecutable) {
    $workspacePython = Join-Path (Split-Path -Parent $rootPath) '.venv/Scripts/python.exe'
    $PythonExecutable = if (Test-Path -LiteralPath $workspacePython -PathType Leaf) {
        $workspacePython
    } else {
        (Get-Command python -ErrorAction Stop).Source
    }
}
Get-Command $PythonExecutable -ErrorAction Stop | Out-Null
$artifactPath = Join-Path $rootPath 'artifacts'
New-Item -ItemType Directory -Path $artifactPath -Force | Out-Null
$runId = 'v{0}-cu128-{1}-{2}-{3}' -f $selectedVersion, (Get-Date).ToUniversalTime().ToString('yyyyMMdd-HHmmss'), $PID, ([Guid]::NewGuid().ToString('N').Substring(0, 8))
$evidencePath = Join-Path $artifactPath "completion-$runId"
New-Item -ItemType Directory -Path $evidencePath | Out-Null
$statusPath = Join-Path $artifactPath 'build-status.json'
$state = [pscustomobject]@{}
if (Test-Path -LiteralPath $statusPath -PathType Leaf) {
    try {
        $previousState = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        # Keep the current build's builder/job limits and root-recorded reuse
        # provenance. Metadata from a different image belongs to its old run.
        if ($previousState.final_image -eq $imageTag) { $state = $previousState }
    } catch {
        Write-Warning "Could not read previous build status; starting a new completion record: $($_.Exception.Message)"
    }
}
# A completion retry must not inherit successful validation/export results or
# an obsolete error from its previous attempt.
foreach ($key in @('status','last_updated_at','completion_error','image_id','image_size_bytes',
                  'image_tar','image_tar_bytes','image_tar_sha256','api_model','api_response',
                  'validation_gpu','nvfp4_validation')) {
    $state.PSObject.Properties.Remove($key)
}
$initialState = @{
    completion_started_at = (Get-Date).ToString('o')
    final_image = $imageTag
    selected_vllm_version = $selectedVersion
    final_image_built = $false
    platform = 'linux/amd64'
    completion_pipeline_pid = $PID
    completion_evidence_directory = $evidencePath
    h200_hardware_tested = $false
    gpu_validation_passed = $false
    nvfp4_validation_passed = $false
    api_validation_passed = $false
    export_validation_passed = $false
}
if (!$state.started_at) { $initialState['started_at'] = (Get-Date).ToString('o') }
foreach ($key in $initialState.Keys) { $state | Add-Member -NotePropertyName $key -NotePropertyValue $initialState[$key] -Force }
$serverName = "vllm-cu128-smoke-$runId"
$serverId = $null
$wakeLock = 0
$transcriptStarted = $false
$pipelineExitCode = 0

function Set-Phase([string]$Phase, [hashtable]$Values = @{}) {
    $Values['status'] = $Phase
    $Values['last_updated_at'] = (Get-Date).ToString('o')
    foreach ($key in $Values.Keys) { $script:state | Add-Member -NotePropertyName $key -NotePropertyValue $Values[$key] -Force }
    $json = $script:state | ConvertTo-Json -Depth 20
    $temporaryStatus = Join-Path $artifactPath "build-status-$PID.tmp"
    [IO.File]::WriteAllText($temporaryStatus, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporaryStatus -Destination $statusPath -Force
    Write-Output "$(Get-Date -Format o) $Phase"
}

function Read-DockerText([string[]]$DockerArguments, [string]$LogName) {
    # Windows PowerShell 5 treats native stderr as error records. Preserve it in
    # its own file, then check the native exit code before processing stdout.
    $ErrorActionPreference = 'Continue'
    $outputLines = & docker @DockerArguments 2> (Join-Path $evidencePath "$LogName.stderr.log")
    $nativeExitCode = $LASTEXITCODE
    if ($nativeExitCode -ne 0) { throw "Docker command failed (exit $nativeExitCode); see $LogName.stderr.log" }
    return (($outputLines | ForEach-Object { [string]$_ }) -join [Environment]::NewLine).Trim()
}

function Invoke-DockerLog([string[]]$DockerArguments, [string]$LogName) {
    $ErrorActionPreference = 'Continue'
    & docker @DockerArguments 2>&1 | Tee-Object -FilePath (Join-Path $evidencePath $LogName) -ErrorAction Stop | Out-Host
    $nativeExitCode = $LASTEXITCODE
    if ($nativeExitCode -ne 0) { throw "Docker command failed (exit $nativeExitCode); see $LogName" }
}

function Remove-OwnedSmokeContainer([switch]$AllowFailure) {
    $ErrorActionPreference = 'Continue'
    # A failed docker run can leave a created container without returning its
    # ID. The unique run label permits cleanup of that container only.
    $inspection = & docker container inspect $serverName 2> (Join-Path $evidencePath 'smoke-cleanup-inspect.stderr.log')
    $inspectExitCode = $LASTEXITCODE
    if ($inspectExitCode -ne 0) {
        if ($script:serverId -and !$AllowFailure) { throw "Cannot inspect this pipeline's smoke container for cleanup" }
        return
    }
    $ownedContainer = @($inspection | ConvertFrom-Json)[0]
    if ($ownedContainer.Config.Labels.'com.lottecard.vllm.completion-run' -ne $runId) {
        throw 'Refusing cleanup: smoke container ownership label does not match this pipeline'
    }
    & docker logs $ownedContainer.Id 2>&1 | Set-Content -LiteralPath (Join-Path $evidencePath 'smoke-server.log') -Encoding utf8 -ErrorAction Stop
    if ($ownedContainer.State.Running) {
        & docker stop $ownedContainer.Id 2>&1 | Out-Host
        if ($LASTEXITCODE -ne 0 -and !$AllowFailure) { throw "Could not stop this pipeline's smoke container" }
    }
    & docker rm $ownedContainer.Id 2>&1 | Out-Host
    if ($LASTEXITCODE -ne 0) {
        if (!$AllowFailure) { throw "Could not remove this pipeline's smoke container" }
        Write-Warning "Smoke container $($ownedContainer.Id) remains; remove this exact container after reviewing its logs."
    } else {
        $script:serverId = $null
    }
}

Push-Location -LiteralPath $rootPath
try {
    Start-Transcript -Path (Join-Path $evidencePath 'completion-pipeline.log') | Out-Null
    $transcriptStarted = $true
    if ($env:OS -eq 'Windows_NT') {
        if (!('VllmCompletionWakeLock' -as [type])) {
            Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class VllmCompletionWakeLock {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
        }
        $wakeLock = [VllmCompletionWakeLock]::SetThreadExecutionState(2147483649)
        if ($wakeLock -eq 0) { throw 'Unable to acquire temporary idle-sleep inhibition' }
    }
    $dockerOs = Read-DockerText @('info','--format','{{.OSType}}') 'docker-info'
    if ($dockerOs -ne 'linux') { throw 'Use the Linux Docker engine (Windows: Docker Desktop with WSL2)' }
    if ($BuildProcessId) {
        $buildProcess = Get-Process -Id $BuildProcessId -ErrorAction Stop
        if ($buildProcess.ProcessName -ne 'docker') { throw 'The supplied build PID is not Docker' }
        $processDetails = Get-CimInstance Win32_Process -Filter "ProcessId = $BuildProcessId"
        $tagPattern = '(?:--tag|-t)(?:=|\s+)"?' + [regex]::Escape($imageTag) + '"?(?:\s|$)'
        if ($processDetails.CommandLine -notmatch '\bbuild(?:x)?\b' -or $processDetails.CommandLine -notmatch $tagPattern) {
            throw 'The supplied Docker PID does not build the selected image tag'
        }
        # Retain the handle before waiting, so a PID reuse cannot be mistaken
        # for the original build and the exit status remains readable.
        $null = $buildProcess.Handle
        if (!$PSBoundParameters.ContainsKey('BuildStartedAt')) { $BuildStartedAt = [DateTimeOffset]$buildProcess.StartTime }
        Set-Phase 'waiting_for_build' @{ build_process_id = $BuildProcessId; build_started_at = $BuildStartedAt.ToString('o') }
        while (!$buildProcess.WaitForExit(30000)) {
            Write-Output "$(Get-Date -Format o) Waiting for Docker build PID $BuildProcessId."
        }
        if ($buildProcess.ExitCode -ne 0) { throw "Docker build failed (exit $($buildProcess.ExitCode)); inspect artifacts/build.log" }
    }

    $inspection = Read-DockerText @('image','inspect',$imageTag) 'image-inspect'
    $imageData = @($inspection | ConvertFrom-Json)[0]
    if ($imageData.Os -ne 'linux' -or $imageData.Architecture -ne 'amd64') { throw 'Unexpected image platform' }
    # Fully cached builds legitimately preserve an older image Created value.
    # Only impose freshness when the caller explicitly requests that cutoff.
    if ($PSBoundParameters.ContainsKey('BuildStartedAt') -and [DateTimeOffset]$imageData.Created -lt $BuildStartedAt) { throw 'Image predates the selected build start time' }
    $inspection | Set-Content -LiteralPath (Join-Path $evidencePath 'image-inspect.json') -Encoding utf8
    Set-Phase 'validating_image' @{ final_image_built = $true; image_id = $imageData.Id; image_size_bytes = $imageData.Size }
    $buildInfoText = Read-DockerText @('run','--rm','--entrypoint','cat',$imageData.Id,'/opt/build-info.json') 'build-info'
    $buildInfo = $buildInfoText | ConvertFrom-Json
    $packageKeys = @{
        vllm='VLLM_VERSION'; torch='TORCH_VERSION'; torchvision='TORCHVISION_VERSION';
        torchaudio='TORCHAUDIO_VERSION'; triton='TRITON_VERSION'; torchcodec='TORCHCODEC_VERSION'
    }
    foreach ($package in $packageKeys.Keys) {
        $suffix = if ($package -eq 'torchcodec') { '+cpu' } elseif ($package -eq 'triton') { '' } else { '+cu128' }
        $expected = $config[$packageKeys[$package]] + $suffix
        if ($buildInfo.packages.$package -ne $expected) { throw "Unexpected $package version: expected $expected, found $($buildInfo.packages.$package)" }
    }
    if ($buildInfo.cuda_base -ne $config['CUDA_BASE'] -or $buildInfo.cuda_toolkit -notmatch 'release 12\.8,') { throw 'CUDA build provenance does not match the configured CUDA 12.8 base' }
    $expectedArches = @($config['TORCH_CUDA_ARCH_LIST'].Split(';') | Sort-Object)
    if ((@($buildInfo.gpu_architectures | Sort-Object) -join ';') -ne ($expectedArches -join ';')) { throw 'GPU architecture provenance does not match configuration' }
    if ((@($buildInfo.required_extensions | Sort-Object) -join ',') -ne (@($config['VLLM_REQUIRED_EXTENSIONS'].Split(',') | Sort-Object) -join ',')) { throw 'Extension provenance does not match configuration' }
    $sourceKeys = @{ vllm='VLLM_COMMIT'; pytorch='TORCH_COMMIT'; torchvision='TORCHVISION_COMMIT'; torchaudio='TORCHAUDIO_COMMIT' }
    foreach ($source in $sourceKeys.Keys) {
        $actualCommit = [string]$buildInfo.source_commits.$source
        if ($actualCommit -notmatch '^[0-9a-fA-F]{40}$') { throw "Missing or invalid $source source commit" }
        if ($config[$sourceKeys[$source]] -and $actualCommit -ne $config[$sourceKeys[$source]]) { throw "$source source commit does not match the configured pin" }
    }
    # A new release may fix one or more of the known source issues upstream.
    # Validate recorded patches rather than requiring a fixed list of IDs.
    if ($null -eq $buildInfo.source_patches -or $null -eq $buildInfo.source_patches.patches) { throw 'Missing source patch provenance' }
    $patchIds = @()
    foreach ($patch in @($buildInfo.source_patches.patches)) {
        if (!$patch.id -or !$patch.source -or $patch.patched_source_sha256 -notmatch '^[0-9a-fA-F]{64}$') { throw 'Invalid source patch provenance entry' }
        if ($patchIds -contains $patch.id) { throw "Duplicate source patch ID: $($patch.id)" }
        if ($patch.helper -and $patch.helper_sha256 -notmatch '^[0-9a-fA-F]{64}$') { throw "Invalid helper provenance for patch $($patch.id)" }
        $patchIds += $patch.id
    }
    $patchText = Read-DockerText @('run','--rm','--entrypoint','cat',$imageData.Id,'/opt/source-patches.json') 'source-patches'
    if ((($patchText | ConvertFrom-Json) | ConvertTo-Json -Depth 20 -Compress) -ne ($buildInfo.source_patches | ConvertTo-Json -Depth 20 -Compress)) { throw 'Embedded source patch records disagree' }
    $frontendText = Read-DockerText @('run','--rm','--entrypoint','cat',$imageData.Id,'/opt/nvidia-cudnn-frontend-wheel.json') 'frontend-wheel'
    $frontendRecord = $frontendText | ConvertFrom-Json
    if ($frontendRecord.code_unchanged -ne $true -or $frontendRecord.original_wheel.sha256 -notmatch '^[0-9a-f]{64}$' -or $frontendRecord.modified_wheel.sha256 -notmatch '^[0-9a-f]{64}$') { throw 'Invalid frontend wheel provenance' }
    if ($frontendRecord.modified_wheel.version -notmatch '\+cu128$' -or $buildInfo.cudnn_frontend.installed_version -ne $frontendRecord.modified_wheel.version) { throw 'Frontend installed version does not match wheel provenance' }
    if (($frontendRecord | ConvertTo-Json -Depth 20 -Compress) -ne ($buildInfo.cudnn_frontend.wheel_provenance | ConvertTo-Json -Depth 20 -Compress)) { throw 'Embedded frontend wheel records disagree' }
    $frontendText | Set-Content -LiteralPath (Join-Path $evidencePath 'frontend-wheel.json') -Encoding utf8
    $buildInfoText | Set-Content -LiteralPath (Join-Path $evidencePath 'build-info.json') -Encoding utf8
    Set-Phase 'validating_gpu' @{ source_commits = $buildInfo.source_commits; patches = $patchIds; gpu_architectures = $buildInfo.gpu_architectures; packages = $buildInfo.packages }
    Invoke-DockerLog @('run','--rm','--gpus','all','--entrypoint','/opt/.venv/bin/python',$imageData.Id,'/opt/verify-stack.py','--gpu') 'gpu-validation.log'
    $gpuProbe = Read-DockerText @('run','--rm','--gpus','all','--entrypoint','/opt/.venv/bin/python',$imageData.Id,'-c',"import json,torch; print(json.dumps(dict(name=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability())))") 'gpu-probe'
    $gpuData = $gpuProbe | ConvertFrom-Json
    $gpuProbe | Set-Content -LiteralPath (Join-Path $evidencePath 'gpu-device.json') -Encoding utf8
    $nvfp4Status = 'not_applicable_to_gpu'
    if ([int]$gpuData.capability[0] -ge 10) {
        if (!(Test-Path -LiteralPath (Join-Path $rootPath 'tests/nvfp4-runtime.py') -PathType Leaf)) { throw 'Missing NVFP4 regression test for this Blackwell GPU' }
        Invoke-DockerLog @('run','--rm','--gpus','all','--mount',"type=bind,source=$rootPath\tests,target=/tests,readonly",'--entrypoint','/opt/.venv/bin/python',$imageData.Id,'/tests/nvfp4-runtime.py') 'nvfp4-validation.log'
        $nvfp4Status = 'passed'
    }
    Set-Phase 'validating_api' @{ gpu_validation_passed = $true; validation_gpu = $gpuData; nvfp4_validation = $nvfp4Status; nvfp4_validation_passed = ($nvfp4Status -eq 'passed') }
    $cachePath = $config['VLLM_CACHE_DIR']
    if (![IO.Path]::IsPathRooted($cachePath)) { $cachePath = Join-Path $rootPath $cachePath }
    $cachePath = (Resolve-Path -LiteralPath $cachePath).Path
    if (Get-NetTCPConnection -LocalPort $SmokePort -State Listen -ErrorAction SilentlyContinue) { throw "Smoke API port $SmokePort is occupied" }
    $serverArguments = @(
        'run','--detach','--init','--name',$serverName,'--label',"com.lottecard.vllm.completion-run=$runId",
        '--gpus','all','--ipc','host','--publish',"127.0.0.1:$($SmokePort):8000",
        '--mount',"type=bind,source=$cachePath,target=/root/.cache/huggingface,readonly",
        '--env','HF_HUB_OFFLINE=1','--env','TRANSFORMERS_OFFLINE=1',
        $imageData.Id,$SmokeModelPath,'--served-model-name',$SmokeModel,'--host','0.0.0.0','--port','8000',
        '--dtype','bfloat16','--attention-backend','FLASH_ATTN','--attention-config.flash_attn_version=2',
        '--enforce-eager','--max-model-len','512','--max-num-seqs','1','--max-num-batched-tokens','512',
        '--gpu-memory-utilization','0.20','--kv-cache-memory-bytes','134217728'
    )
    $serverId = Read-DockerText $serverArguments 'smoke-start'
    if ($serverId -notmatch '^[0-9a-f]{64}$') { throw 'Smoke server did not return a valid container ID' }
    $apiReportPath = Join-Path $evidencePath 'smoke-api.json'
    $ErrorActionPreference = 'Continue'
    & $PythonExecutable (Join-Path $rootPath 'scripts/smoke-api.py') --url "http://127.0.0.1:$SmokePort" --model $SmokeModel --output $apiReportPath 2>&1 | Tee-Object -FilePath (Join-Path $evidencePath 'smoke-api-client.log') -ErrorAction Stop | Out-Host
    $apiExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    Remove-OwnedSmokeContainer
    if ($apiExitCode -ne 0) { throw 'API inference failed; inspect smoke-api.json and smoke-server.log' }
    $apiReport = Get-Content -LiteralPath $apiReportPath -Raw | ConvertFrom-Json
    if ($apiReport.passed -ne $true -or $apiReport.checks.chat_completion.passed -ne $true) { throw 'Smoke API report does not confirm successful inference' }

    $tarPath = Join-Path $artifactPath "vllm-v$selectedVersion-cu128-linux-amd64.tar"
    $partialPath = Join-Path $evidencePath "vllm-v$selectedVersion-cu128-linux-amd64.partial.tar"
    if (Test-Path -LiteralPath $tarPath) { throw "Export path already exists; preserved without overwrite: $tarPath" }
    Set-Phase 'exporting_image' @{ api_validation_passed = $true; api_model = $SmokeModel; api_response = $apiReport.checks.chat_completion.assistant_content }
    # Export by tag so docker load also restores the selected repository/tag.
    $currentId = Read-DockerText @('image','inspect',$imageTag,'--format','{{.Id}}') 'export-tag-check'
    if ($currentId -ne $imageData.Id) { throw 'Image tag changed during validation' }
    Invoke-DockerLog @('save','--output',$partialPath,$imageTag) 'image-export.log'
    # Docker's containerd store may identify an image by manifest/index digest.
    # Authenticate that graph and its Linux/amd64 config/layers before renaming.
    $ErrorActionPreference = 'Continue'
    & $PythonExecutable (Join-Path $rootPath 'scripts/verify-image-export.py') $partialPath $imageTag $imageData.Id --output (Join-Path $evidencePath 'image-export-validation.json') 2>&1 | Tee-Object -FilePath (Join-Path $evidencePath 'image-export-validation.log') -ErrorAction Stop | Out-Host
    $tarCheckExitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($tarCheckExitCode -ne 0) { throw 'Exported archive failed manifest/config validation; partial archive preserved' }
    # Move without -Force: do not replace an export created by another process.
    Move-Item -LiteralPath $partialPath -Destination $tarPath
    Set-Phase 'hashing_export'
    $hash = (Get-FileHash -LiteralPath $tarPath -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $([IO.Path]::GetFileName($tarPath))" | Set-Content -LiteralPath ($tarPath + '.sha256') -Encoding ascii
    Set-Phase 'built_validated_exported' @{
        image_tar = $tarPath; image_tar_bytes = (Get-Item -LiteralPath $tarPath).Length
        image_tar_sha256 = $hash; export_validation_passed = $true
    }
} catch {
    $pipelineError = $_
    $pipelineExitCode = 1
    try { Set-Phase 'completion_pipeline_failed' @{ completion_error = $pipelineError.Exception.Message } }
    catch { Write-Warning "Could not update build status: $($_.Exception.Message)" }
    Write-Error -Message $pipelineError -ErrorAction Continue
} finally {
    try { Remove-OwnedSmokeContainer -AllowFailure }
    catch {
        $cleanupError = $_.Exception.Message
        Write-Warning "Smoke cleanup failed: $cleanupError"
        $pipelineExitCode = 1
        try { Set-Phase 'completion_pipeline_failed' @{ completion_error = "Smoke cleanup failed: $cleanupError" } }
        catch { Write-Warning "Could not update build status: $($_.Exception.Message)" }
    }
    if ($wakeLock -ne 0) { [VllmCompletionWakeLock]::SetThreadExecutionState(2147483648) | Out-Null }
    if ($transcriptStarted) { Stop-Transcript -ErrorAction Continue | Out-Null }
    Pop-Location
}
exit $pipelineExitCode
