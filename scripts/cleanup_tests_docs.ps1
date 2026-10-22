[CmdletBinding()]
param([switch]$Execute)

# Read-only preview by default; -Execute removes only the audited test/doc targets.
$ErrorActionPreference = 'Stop'
$cleanupRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($cleanupRoot -ne 'D:\code\HOI') { throw "Unexpected project root: $cleanupRoot" }
$reportPath = Join-Path $cleanupRoot 'docs/cleanup_tests_docs_20261022.json'
$report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
$allowedTargets = @(
    'tests', 'scripts/verify_token_ar.py', 'scripts/cleanup_current_model.ps1',
    'docs/cleanup_file_list_20261021.txt', 'docs/code_cleanup_20261021.json',
    'docs/data_cleanup_20261021.json', 'docs/TOKEN_AR_VALIDATION.json',
    'docs/TOKEN_AR_QWEN35_VALIDATION.json',
    'runs/token_ar/verification_20261022', 'runs/token_ar/verification_final_20261022',
    'runs/token_ar/verification_qwen35_20261022', 'runs/token_ar/verification_qwen3_regression_20261022',
    'runs/token_ar/qwen35_unit_tests.log', 'runs/token_ar/verification_qwen35_20261022.log',
    'runs/token_ar/verification_qwen3_regression_20261022.log', '.ruff_cache'
)
$actualTargets = @($report.deletions | ForEach-Object { $_.path })
if ($actualTargets.Count -ne $allowedTargets.Count -or
    @($actualTargets | Select-Object -Unique).Count -ne $allowedTargets.Count -or
    @(Compare-Object ($allowedTargets | Sort-Object) ($actualTargets | Sort-Object)).Count) {
    throw 'The fixed cleanup target list changed.'
}
if ($report.validation.runtime_cli_help_passed -ne 12) { throw 'Runtime CLI validation is incomplete.' }

function Assert-CleanupTarget([string]$relative) {
    if ($allowedTargets -notcontains $relative) { throw "Unapproved target: $relative" }
    $target = [IO.Path]::GetFullPath((Join-Path $cleanupRoot $relative))
    if (-not $target.StartsWith($cleanupRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside project: $target"
    }
    foreach ($name in @('external', 'OMOMO', 'data', 'src', 'configs', '.git', '.venv')) {
        $protected = Join-Path $cleanupRoot $name
        if ($target.Equals($protected, [StringComparison]::OrdinalIgnoreCase) -or
            $target.StartsWith($protected + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Protected path: $target"
        }
    }
    if (Test-Path -LiteralPath $target) {
        $item = Get-Item -LiteralPath $target -Force
        $ancestor = $item
        while ($null -ne $ancestor) {
            if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) {
                throw "Reparse path refused: $($ancestor.FullName)"
            }
            if ($ancestor.FullName -eq $cleanupRoot) { break }
            $ancestor = Get-Item -LiteralPath (Split-Path -Parent $ancestor.FullName) -Force
        }
        if ($item.PSIsContainer -and @(Get-ChildItem -LiteralPath $target -Recurse -Force |
            Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
            throw "Reparse child refused: $target"
        }
    }
    return $target
}

$verify = @'
import ast, hashlib, json, sys
from pathlib import Path
r = json.loads(Path('docs/cleanup_tests_docs_20261022.json').read_text(encoding='utf-8-sig'))
def inventory(p):
    p = Path(p)
    files = sorted(f for f in p.rglob('*') if f.is_file()) if p.is_dir() else [p]
    rows = [(f.as_posix(), f.stat().st_size, f.stat().st_mtime_ns) for f in files]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()
for folder, expected in r['protected_inventory'].items():
    assert inventory(folder) == expected['inventory_sha256'], 'Protected data changed: ' + folder
for item in r['protected_code'] + r['retained_documents_and_tools']:
    p = Path(item['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], 'Reviewed file changed: ' + str(p)
    if p.suffix == '.py':
        ast.parse(p.read_text(encoding='utf-8-sig'))
for e in r['deletions']:
    p = Path(e['path'])
    if sys.argv[1] == 'post':
        assert not p.exists(), 'Deletion target remains: ' + str(p)
    elif not (e['deleted'] and not p.exists()):
        assert inventory(p) == e['inventory_sha256'], 'Deletion target changed: ' + str(p)
print('Both model sources/configs, current docs, data, body assets, raw OMOMO and external verified.', flush=True)
'@
$python = Join-Path $cleanupRoot '.venv/Scripts/python.exe'
Push-Location -LiteralPath $cleanupRoot
try {
    foreach ($entry in $report.deletions) { $null = Assert-CleanupTarget $entry.path }
    $report.deletions | Select-Object path, files, bytes, deleted | Format-Table -AutoSize
    $verify | & $python -B - pre
    if ($LASTEXITCODE -ne 0) { throw 'Pre-deletion verification failed.' }
    if (-not $Execute) {
        Write-Output ('Read-only preview passed: {0} files, {1:N2} MB. Add -Execute to delete.' -f $report.planned_files, ($report.planned_bytes / 1000000))
        return
    }
    $report.status = 'deleting'
    try {
        foreach ($entry in $report.deletions) {
            $target = Assert-CleanupTarget $entry.path
            if ($entry.deleted -and -not (Test-Path -LiteralPath $target)) { continue }
            Remove-Item -LiteralPath $target -Recurse -Force
            if (Test-Path -LiteralPath $target) { throw "Target remains: $target" }
            $entry.deleted = $true
            $report | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $reportPath -Encoding UTF8
            Write-Output "Deleted $($entry.path)"
        }
        $verify | & $python -B - post
        if ($LASTEXITCODE -ne 0) { throw 'Post-deletion verification failed.' }
        'from setuptools import setup; setup(script_args=["egg_info"])' | & $python -B -
        if ($LASTEXITCODE -ne 0) { throw 'Package metadata refresh failed.' }
        $report.status = 'complete'
        $report | Add-Member -NotePropertyName post_cleanup_verified -NotePropertyValue $true -Force
    } catch {
        $report.status = 'partial_failure'
        throw
    } finally {
        $done = @($report.deletions | Where-Object { $_.deleted })
        $report.deleted_files = [long](($done | Measure-Object -Property files -Sum).Sum)
        $report.deleted_bytes = [long](($done | Measure-Object -Property bytes -Sum).Sum)
        $report | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $reportPath -Encoding UTF8
    }
    Write-Output ('Deleted {0} files, {1:N2} MB. Both model paths and external are preserved.' -f $report.deleted_files, ($report.deleted_bytes / 1000000))
} finally {
    Pop-Location
}
