# Parse configuration as data. Never dot-source or evaluate an .env file.
function Get-VllmBuildKeys {
    @('VLLM_VERSION', 'TORCH_VERSION', 'TORCHVISION_VERSION', 'TORCHAUDIO_VERSION',
      'TRITON_VERSION', 'TORCHCODEC_VERSION', 'RUST_VERSION', 'CUDA_BASE',
      'VLLM_COMMIT', 'TORCH_COMMIT', 'TORCHVISION_COMMIT', 'TORCHAUDIO_COMMIT',
      'VLLM_REQUIRED_EXTENSIONS', 'TORCH_CUDA_ARCH_LIST', 'MAX_JOBS',
      'TORCH_MAX_JOBS', 'VLLM_MAX_JOBS', 'NVCC_THREADS')
}

function Read-VllmConfigFile {
    param([string]$Path, [System.Collections.IDictionary]$Config)
    $lineNumber = 0
    $seenKeys = @{}
    foreach ($rawLine in [System.IO.File]::ReadAllLines($Path)) {
        $lineNumber++
        $line = $rawLine.Trim()
        if (!$line -or $line.StartsWith('#')) { continue }
        if ($line -cnotmatch '^([A-Z][A-Z0-9_]*)\s*=(.*)$') {
            throw "${Path}:${lineNumber}: expected KEY=VALUE"
        }
        $key = $Matches[1]
        $value = $Matches[2].Trim()
        if (!$Config.Contains($key)) { throw "${Path}:${lineNumber}: unknown setting $key" }
        if ($seenKeys.ContainsKey($key)) { throw "${Path}:${lineNumber}: duplicate setting $key" }
        $seenKeys[$key] = $true
        if ($value.StartsWith('"') -or $value.StartsWith("'")) {
            if ($value.Length -lt 2 -or $value[$value.Length - 1] -ne $value[0]) {
                throw "${Path}:${lineNumber}: unmatched quote"
            }
            $value = $value.Substring(1, $value.Length - 2)
        }
        $Config[$key] = $value
    }
}

function Get-VllmConfig {
    param([Parameter(Mandatory)][string]$Root, [hashtable]$Overrides = @{})
    $config = [ordered]@{}
    foreach ($key in (Get-VllmBuildKeys)) { $config[$key] = '' }
    $config['VLLM_IMAGE_REPOSITORY'] = ''
    $config['VLLM_IMAGE'] = ''
    $config['VLLM_MODEL'] = ''
    $config['VLLM_PORT'] = '8000'
    $config['VLLM_CONTAINER_NAME'] = 'vllm-cu128'
    $config['VLLM_CACHE_DIR'] = Join-Path $Root '.cache/huggingface'
    $config['VLLM_MODELS_DIR'] = ''
    $config['HF_TOKEN'] = ''
    Read-VllmConfigFile -Path (Join-Path $Root 'versions.env') -Config $config
    $runtimeFile = Join-Path $Root '.env'
    if (Test-Path -LiteralPath $runtimeFile -PathType Leaf) {
        Read-VllmConfigFile -Path $runtimeFile -Config $config
    }
    foreach ($key in @($config.Keys)) {
        $environmentValue = [Environment]::GetEnvironmentVariable($key, 'Process')
        if ($null -ne $environmentValue) { $config[$key] = $environmentValue }
    }
    foreach ($key in $Overrides.Keys) {
        if (!$config.Contains($key)) { throw "Unknown setting $key" }
        $config[$key] = [string]$Overrides[$key]
    }
    foreach ($key in @('VLLM_VERSION','TORCH_VERSION','TORCHVISION_VERSION','TORCHAUDIO_VERSION',
                       'TRITON_VERSION','TORCHCODEC_VERSION')) {
        if ($config[$key] -cnotmatch '^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+|\.post\d+)?$') {
            throw "Invalid $key release version: $($config[$key])"
        }
    }
    if ($config['RUST_VERSION'] -cnotmatch '^\d+\.\d+(?:\.\d+)?$') {
        throw 'RUST_VERSION must be a numeric release such as 1.95 or 1.95.0'
    }
    foreach ($key in @('VLLM_COMMIT','TORCH_COMMIT','TORCHVISION_COMMIT','TORCHAUDIO_COMMIT')) {
        if ($config[$key] -and $config[$key] -notmatch '^[0-9a-fA-F]{40}$') {
            throw "$key must be empty or a full 40-character commit SHA"
        }
    }
    foreach ($key in @('MAX_JOBS','TORCH_MAX_JOBS','VLLM_MAX_JOBS','NVCC_THREADS','VLLM_PORT')) {
        if ($config[$key] -notmatch '^[1-9][0-9]*$') { throw "$key must be a positive integer" }
        $limit = if ($key -eq 'VLLM_PORT') { 65535 } else { 1024 }
        if ([long]$config[$key] -gt $limit) { throw "$key must not exceed $limit" }
    }
    if ($config['TORCH_CUDA_ARCH_LIST'] -cnotmatch '^\d+\.\d+[a-z]?(?:\+PTX)?(?:;\d+\.\d+[a-z]?(?:\+PTX)?)*$') {
        throw 'TORCH_CUDA_ARCH_LIST must contain semicolon-separated CUDA architectures'
    }
    if ($config['VLLM_REQUIRED_EXTENSIONS'] -cnotmatch '^[A-Za-z_][A-Za-z0-9_]*(?:,[A-Za-z_][A-Za-z0-9_]*)*$') {
        throw 'VLLM_REQUIRED_EXTENSIONS must contain comma-separated module names'
    }
    foreach ($key in @('CUDA_BASE', 'VLLM_IMAGE_REPOSITORY', 'VLLM_CACHE_DIR', 'VLLM_CONTAINER_NAME')) {
        if ([string]::IsNullOrWhiteSpace($config[$key])) { throw "$key must not be empty" }
    }
    if (!$config['VLLM_IMAGE']) {
        $config['VLLM_IMAGE'] = '{0}:v{1}-cu128' -f $config['VLLM_IMAGE_REPOSITORY'], $config['VLLM_VERSION']
    }
    return $config
}

function Write-VllmConfig {
    param([System.Collections.IDictionary]$Config)
    foreach ($key in $Config.Keys) {
        $value = if ($key -eq 'HF_TOKEN' -and $Config[$key]) { '[redacted]' } else { $Config[$key] }
        Write-Output "$key=$value"
    }
}
