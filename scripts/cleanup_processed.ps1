[CmdletBinding()]
param([switch]$Execute)

# Manual cleanup of audited, superseded processed data. No raw data is targeted.
# Default: read-only preview. -Execute deletes the exact audited targets.
$ErrorActionPreference = 'Stop'
$cleanupRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($cleanupRoot -ne 'D:\code\HOI') { throw "Unexpected project root: $cleanupRoot" }
$reportPath = Join-Path $cleanupRoot 'docs/processed_cleanup_report.json'
$report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
$allowedTargets = @(
    'data/processed/behave', 'data/processed/omomo_train', 'data/processed/omomo_test',
    'data/processed/behave_with_mesh.jsonl', 'data/processed/behave_with_text.jsonl',
    'data/processed/omomo_combined.jsonl', 'data/processed/omomo_with_mesh.jsonl',
    'data/processed/omomo_with_text.jsonl', 'data/processed/omomo_text_verification.json',
    'data/processed/behave_with_body_fk.report.json', 'data/processed/omomo_with_body_fk.report.json'
)
if ($report.deletions.Count -ne $allowedTargets.Count) { throw 'Cleanup plan changed.' }
if (-not $report.validation.cli_and_python_syntax_passed -or
    $report.validation.motion_records.omomo -ne 4736 -or
    $report.validation.motion_records.behave -ne 1454) {
    throw 'Full runtime-data validation has not completed.'
}
foreach ($rootPath in @($cleanupRoot, (Join-Path $cleanupRoot 'data'), (Join-Path $cleanupRoot 'data/processed'))) {
    if ((Get-Item -LiteralPath $rootPath -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
        throw "Reparse root refused: $rootPath"
    }
}
foreach ($entry in $report.deletions) {
    if ($allowedTargets -notcontains $entry.path) { throw "Unapproved target: $($entry.path)" }
    $target = [IO.Path]::GetFullPath((Join-Path $cleanupRoot $entry.path))
    if (-not $target.StartsWith($cleanupRoot + '\data\processed\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside processed: $target"
    }
    if (-not (Test-Path -LiteralPath $target)) {
        if ($entry.deleted) { continue }
        throw "Target changed since audit: $target"
    }
    $item = Get-Item -LiteralPath $target -Force
    $contents = @($item)
    if ($item.PSIsContainer) { $contents += @(Get-ChildItem -LiteralPath $target -Recurse -Force) }
    if (@($contents | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) {
        throw "Reparse point in target: $target"
    }
}
$report.deletions | Select-Object path, files, bytes, deleted | Format-Table -AutoSize
Write-Output ('Delete: {0:N2} MB; retain runtime data: {1:N2} MB.' -f ($report.planned_bytes / 1000000), ($report.retained_bytes / 1000000))
$verifyData = @'
import hashlib, json, sys
from pathlib import Path
root = Path.cwd().resolve()
r = json.loads((root / 'docs/processed_cleanup_report.json').read_text(encoding='utf-8-sig'))
def inventory(paths):
    rows = []
    for p in paths:
        files = [f for f in p.rglob('*') if f.is_file()] if p.is_dir() else [p]
        for f in files:
            s = f.stat()
            rows.append((f.relative_to(root).as_posix(), s.st_size, s.st_mtime_ns))
    rows.sort()
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()
keep = [root / 'data/processed' / n for n in r['retained_top_level']]
assert inventory(keep) == r['retained_inventory']['path_size_mtime_sha256'], 'Final data changed since validation'
for name, expected in r['protected_inventory'].items():
    assert inventory([root / name]) == expected['path_size_mtime_sha256'], 'Protected assets changed: ' + name
for saved in r['preserved_conversion_reports']:
    p = root / saved['destination']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == saved['sha256'], 'Preserved report mismatch: ' + str(p)
for dataset in ('omomo', 'behave'):
    manifest = root / 'data/processed' / (dataset + '_with_body_fk.jsonl')
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        refs = [row[k] for k in ('path', 'text_features_path', 'object_mesh_path') if row.get(k)]
        refs += [v['text_features_path'] for v in row.get('text_variants', []) if v.get('text_features_path')]
        for ref in refs:
            p = (manifest.parent / ref).resolve()
            assert p.is_file() and any(p == q or p.is_relative_to(q) for q in keep), 'Unexpected runtime dependency: ' + str(p)
if sys.argv[1] == 'pre':
    for entry in r['deletions']:
        p = root / entry['path']
        if entry['deleted'] and not p.exists():
            continue
        assert inventory([p]) == entry['path_size_mtime_sha256'], 'Deletion target changed: ' + str(p)
else:
    assert all(not (root / e['path']).exists() for e in r['deletions']), 'Deletion targets remain'
    actual = {p.name for p in (root / 'data/processed').iterdir()}
    assert actual == set(r['retained_top_level']), 'Unexpected remaining processed entry: ' + str(actual)
print('Final motion, text, meshes, raw data, body models and reports verified; runtime references are complete.')
'@
$python = Join-Path $cleanupRoot '.venv/Scripts/python.exe'
Push-Location -LiteralPath $cleanupRoot
try {
    $verifyData | & $python -B - pre
    if ($LASTEXITCODE -ne 0) { throw 'Pre-deletion data verification failed.' }
    if (-not $Execute) {
        Write-Output 'Read-only preview. Add -Execute to delete the audited intermediate data.'
        return
    }
    $report.status = 'deleting'
    try {
        foreach ($entry in $report.deletions) {
            $target = [IO.Path]::GetFullPath((Join-Path $cleanupRoot $entry.path))
            if ($entry.deleted -and -not (Test-Path -LiteralPath $target)) { continue }
            Remove-Item -LiteralPath $target -Recurse -Force
            if (Test-Path -LiteralPath $target) { throw "Target remains: $target" }
            $entry.deleted = $true
            $report | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $reportPath -Encoding UTF8
        }
        $verifyData | & $python -B - post
        if ($LASTEXITCODE -ne 0) { throw 'Post-deletion data verification failed.' }
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
    Write-Output ('Deleted {0} files, {1:N2} MB. Processed now contains four directories and two manifests.' -f $report.deleted_files, ($report.deleted_bytes / 1000000))
} finally {
    Pop-Location
}
