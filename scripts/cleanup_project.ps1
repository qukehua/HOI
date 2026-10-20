[CmdletBinding()]
param([switch]$Execute)

# Prepared for manual review and execution. This script was not executed by Codex:
# the execution policy rejected both batch and single-file deletion commands.
# Without -Execute it only validates and displays the exact cleanup plan.
$ErrorActionPreference = 'Stop'
$cleanupRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($cleanupRoot -ne 'D:\code\HOI') { throw "Unexpected project root: $cleanupRoot" }
$reportPath = Join-Path $cleanupRoot 'docs/project_cleanup_report.json'
$report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
$allowedTargets = @(
    'tests', 'regression_tests', 'configs/smoke.yaml', 'scripts/smoke_pipeline.py',
    'runs', 'tmp', '.pytest_cache', '.ruff_cache', 'docs/data_cleanup_report.json',
    'OMOMO.rar', 'src/unified_hoi/__pycache__',
    'src/unified_hoi/integrations/__pycache__', 'scripts/__pycache__',
    'external/tridi/tridi/model/denoising/__pycache__'
)
if ($report.duplicate_archive_checks.Count -ne 4 -or
    @($report.duplicate_archive_checks | Where-Object { -not $_.passed }).Count -ne 0) {
    throw 'Duplicate archive verification is incomplete.'
}
if ($report.deletions.Count -ne $allowedTargets.Count) { throw 'Cleanup plan changed.' }
if ((Get-Item -LiteralPath $cleanupRoot -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Project root is a reparse point.'
}

foreach ($entry in $report.deletions) {
    if ($allowedTargets -notcontains $entry.path) { throw "Unapproved target: $($entry.path)" }
    $target = [IO.Path]::GetFullPath((Join-Path $cleanupRoot $entry.path))
    if (-not $target.StartsWith($cleanupRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside project: $target"
    }
    if (-not (Test-Path -LiteralPath $target)) {
        if ($entry.deleted) { continue }
        throw "Target changed since inventory: $target"
    }
    $item = Get-Item -LiteralPath $target -Force
    $contents = @($item)
    if ($item.PSIsContainer) { $contents += @(Get-ChildItem -LiteralPath $target -Force -Recurse) }
    if (@($contents | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw "Reparse point in target: $target"
    }
    $files = @($contents | Where-Object { -not $_.PSIsContainer })
    $bytes = [long](($files | Measure-Object -Property Length -Sum).Sum)
    if ($files.Count -ne $entry.files -or $bytes -ne $entry.bytes) {
        throw "Target changed since inventory; inspect before deleting: $target"
    }
}

$report.deletions | Select-Object path, files, bytes, deleted | Format-Table -AutoSize
Write-Output ('Planned size: {0:N3} GB' -f ($report.planned_bytes / 1000000000))
if (-not $Execute) {
    Write-Output 'Read-only preview. Add -Execute to delete these exact targets.'
    return
}

# Recheck all retained data paths, sizes and modification times before relying on
# the archive comparisons. Any data change requires a new audit instead of deletion.
$python = Join-Path $cleanupRoot '.venv/Scripts/python.exe'
$assetCheck = @'
import hashlib, json
from pathlib import Path
root = Path.cwd().resolve()
report = json.loads((root / 'docs/project_cleanup_report.json').read_text(encoding='utf-8-sig'))
for name, expected in report['preserved_assets_before'].items():
    rows = []
    for p in (root / name).rglob('*'):
        if p.is_file():
            s = p.stat()
            rows.append((p.relative_to(root).as_posix(), s.st_size, s.st_mtime_ns))
    rows.sort()
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    if digest != expected['path_size_mtime_sha256']:
        raise SystemExit('Retained assets changed since audit: ' + name)
print('Retained data and model inventory matches the verified archive audit.')
'@
Push-Location -LiteralPath $cleanupRoot
try {
    $assetCheck | & $python -B -
    if ($LASTEXITCODE -ne 0) { throw 'Retained asset verification failed.' }
    $report.status = 'deleting'
    $report | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $reportPath -Encoding UTF8
    try {
        foreach ($entry in $report.deletions) {
            $target = [IO.Path]::GetFullPath((Join-Path $cleanupRoot $entry.path))
            if ($entry.deleted -and -not (Test-Path -LiteralPath $target)) { continue }
            # All recursive targets and their descendants were checked above.
            Remove-Item -LiteralPath $target -Recurse -Force
            if (Test-Path -LiteralPath $target) { throw "Target remains: $target" }
            $entry.deleted = $true
            $report | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $reportPath -Encoding UTF8
        }
        $report.status = 'complete'
    } catch {
        $report.status = 'partial_failure'
        throw
    } finally {
        $done = @($report.deletions | Where-Object { $_.deleted })
        $report.deleted_files = [long](($done | Measure-Object -Property files -Sum).Sum)
        $report.deleted_bytes = [long](($done | Measure-Object -Property bytes -Sum).Sum)
        $report | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $reportPath -Encoding UTF8
    }
    Write-Output ('Deleted {0} files, {1:N3} GB.' -f $report.deleted_files, ($report.deleted_bytes / 1000000000))
    & $python -B -c "from setuptools import setup; setup(script_args=['egg_info'])"
    if ($LASTEXITCODE -ne 0) { Write-Warning 'Files were deleted, but package metadata refresh failed.' }
} finally {
    Pop-Location
}
