$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$downloads = Join-Path $root 'data\downloads'
$curl = (Get-Command curl.exe).Source
$tar = (Get-Command tar.exe).Source
New-Item -ItemType Directory -Path $downloads -Force | Out-Null

$datasets = @(
    @{
        Name = 'refcoco'
        Destination = 'data\raw\refer'
        Required = @('refcoco\instances.json', 'refcoco\refs(unc).p')
        Urls = @(
            'https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip',
            'https://web.archive.org/web/20220413011718id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip'
        )
    }
    @{
        Name = 'refcocog'
        Destination = 'data\raw\refer'
        Required = @('refcocog\instances.json', 'refcocog\refs(google).p')
        Urls = @(
            'https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip',
            'https://web.archive.org/web/20220413012904id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip'
        )
    }
    @{
        Name = 'train2014'
        Destination = 'data\raw\coco'
        Required = @('train2014\COCO_train2014_000000000009.jpg')
        ImageCount = 82783
        Sha256 = 'ede4087e640bddba550e090eae701092534b554b42b05ac33f0300b984b31775'
        Urls = @('http://images.cocodataset.org/zips/train2014.zip', 'https://images.cocodataset.org/zips/train2014.zip')
    }
    @{
        Name = 'annotations_trainval2014'
        Destination = 'data\raw\coco'
        Required = @('annotations\instances_train2014.json', 'annotations\instances_val2014.json')
        Sha256 = '031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009'
        Urls = @('http://images.cocodataset.org/annotations/annotations_trainval2014.zip', 'https://images.cocodataset.org/annotations/annotations_trainval2014.zip')
    }
)

function Test-Dataset {
    param([hashtable]$Dataset, [string]$Destination)

    foreach ($relativePath in $Dataset.Required) {
        $path = Join-Path $Destination $relativePath
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            return $false
        }
        if ((Get-Item -LiteralPath $path).Length -eq 0) {
            return $false
        }
    }
    if ($Dataset.ContainsKey('ImageCount')) {
        $imageDirectory = Join-Path $Destination 'train2014'
        return ([System.IO.Directory]::GetFiles($imageDirectory, '*.jpg').Length -eq $Dataset.ImageCount)
    }
    return $true
}

function Install-Dataset {
    param([hashtable]$Dataset)

    $name = $Dataset.Name
    $destination = Join-Path $root $Dataset.Destination
    $archive = Join-Path $downloads "$name.zip"
    $marker = Join-Path $downloads "$name.extracted"
    New-Item -ItemType Directory -Path $destination -Force | Out-Null

    if ((Test-Path -LiteralPath $marker) -and (Test-Dataset $Dataset $destination)) {
        Write-Host "[OK] $name already extracted"
        return
    }

    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
        for ($index = 0; $index -lt $Dataset.Urls.Count; $index++) {
            $url = $Dataset.Urls[$index]
            $partial = Join-Path $downloads "$name.source$index.part"
            $curlOptions = @('--fail', '--location', '--retry', '2', '--connect-timeout', '15')
            Write-Host "[DOWNLOAD] $name"
            & $curl @curlOptions --continue-at - --output $partial $url
            if (($LASTEXITCODE -in @(22, 33, 36)) -and (Test-Path -LiteralPath $partial)) {
                # A failed resume can also occur when the partial file is already complete.
                & $tar -tf $partial | Out-Null
                if ($LASTEXITCODE -ne 0) {
                    Remove-Item -LiteralPath $partial
                    & $curl @curlOptions --output $partial $url
                }
            }
            if ($LASTEXITCODE -ne 0) {
                continue
            }
            if ($Dataset.ContainsKey('Sha256')) {
                $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
                if ($actual -ne $Dataset.Sha256) {
                    Write-Warning "SHA-256 mismatch for $name from $url"
                    Remove-Item -LiteralPath $partial -Force
                    continue
                }
            }
            & $tar -tf $partial | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Remove-Item -LiteralPath $partial -Force
                continue
            }
            Move-Item -LiteralPath $partial -Destination $archive
            break
        }
    }

    if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
        throw "Could not download $name.zip. Rerun the downloader or place the original ZIP in $downloads."
    }
    if ($Dataset.ContainsKey('Sha256')) {
        $actual = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -ne $Dataset.Sha256) { throw "SHA-256 mismatch for existing archive $archive" }
    }
    Write-Host "[EXTRACT] $name"
    & $tar -xf $archive -C $destination
    if ($LASTEXITCODE -ne 0 -or -not (Test-Dataset $Dataset $destination)) {
        throw "Extraction or verification failed for $archive. Check free space and validate the archive."
    }
    Set-Content -LiteralPath $marker -Value 'verified' -Encoding Ascii
    Write-Host "[OK] $name extracted and verified"
}

$failed = @()
foreach ($dataset in $datasets) {
    try {
        Install-Dataset $dataset
    }
    catch {
        $failed += $dataset.Name
        Write-Warning $_.Exception.Message
    }
}
if ($failed.Count -gt 0) {
    throw ("Dataset setup incomplete: " + ($failed -join ', '))
}
Write-Host '[DONE] All four dataset archives are ready.'
