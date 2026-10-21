[CmdletBinding()]
param([switch]$Execute)

# Preview by default. Delete only the fixed, audited plan when -Execute is supplied.
$ErrorActionPreference = 'Stop'
$cleanupRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
if ($cleanupRoot -ne 'D:\code\HOI') { throw "Unexpected project root: $cleanupRoot" }
$protectedExternal = Join-Path $cleanupRoot 'external'
$reportPath = Join-Path $cleanupRoot 'docs/code_cleanup_20261021.json'
$report = Get-Content -LiteralPath $reportPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($report.validation.unit_tests_after -ne 13 -or
    $report.validation.imports_without_retired_modules -ne 22 -or
    -not $report.validation.prepared_datasets_and_codecs_checked) {
    throw 'Runtime verification has not completed.'
}
if ((Get-Item -LiteralPath $cleanupRoot -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Reparse project root refused.'
}
foreach ($entry in $report.deletions) {
    $target = [IO.Path]::GetFullPath((Join-Path $cleanupRoot $entry.path))
    if (-not $target.StartsWith($cleanupRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target outside project: $target"
    }
    if ($target.Equals($protectedExternal, [StringComparison]::OrdinalIgnoreCase) -or
        $target.StartsWith($protectedExternal + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "User-protected external directory: $target"
    }
    if (-not (Test-Path -LiteralPath $target)) {
        if ($entry.deleted) { continue }
        throw "Target changed since audit: $target"
    }
    $ancestor = Get-Item -LiteralPath $target -Force
    while ($ancestor.FullName -ne $cleanupRoot) {
        if ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Reparse path refused: $($ancestor.FullName)"
        }
        $ancestor = Get-Item -LiteralPath (Split-Path -Parent $ancestor.FullName) -Force
    }
    $item = Get-Item -LiteralPath $target -Force
    if ($item.PSIsContainer) {
        $links = @(Get-ChildItem -LiteralPath $target -Recurse -Force |
            Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint })
        if ($links.Count) { throw "Reparse child refused: $target" }
    }
}

$verify = @'
import ast, hashlib, importlib, importlib.abc, json, sys
from pathlib import Path
root = Path.cwd().resolve()
r = json.loads(Path('docs/code_cleanup_20261021.json').read_text(encoding='utf-8-sig'))
paths = sorted(e['path'] for e in r['deletions'])
assert len(paths) == len(set(paths))
assert hashlib.sha256(json.dumps(paths).encode()).hexdigest() == '918cbafc5aa3f4eccb4ddff1156932efe66615908a92cfbb861b880755021953', 'Cleanup target list changed'
rows = []
for folder in r['runtime_directories']:
    for p in sorted(Path(folder).rglob('*')):
        if p.is_file():
            rows.append((p.as_posix(), p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()))
assert hashlib.sha256(json.dumps(rows).encode()).hexdigest() == r['runtime_content_fingerprint'], 'Current data/assets changed'
raw = sorted((p.as_posix(), p.stat().st_size, p.stat().st_mtime_ns) for p in Path('OMOMO').rglob('*') if p.is_file())
assert hashlib.sha256(json.dumps(raw).encode()).hexdigest() == r['protected_omomo_inventory'], 'Raw OMOMO changed'
for item in r['retained_sources']:
    p = Path(item['path'])
    assert hashlib.sha256(p.read_bytes()).hexdigest() == item['sha256'], 'Reviewed source changed: ' + str(p)
targets = [root / e['path'] for e in r['deletions']]
def targeted(p):
    p = p.resolve()
    return any(p == t or t in p.parents for t in targets)
removed = set(r['retired_modules'])
class AbsentLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in removed:
            raise ModuleNotFoundError('Retired module imported: ' + fullname)
sys.meta_path.insert(0, AbsentLegacy())
sys.path.insert(0, str(root / 'src'))
for item in r['retained_sources']:
    p = Path(item['path'])
    if p.suffix == '.py':
        ast.parse(p.read_text(encoding='utf-8-sig'))
    if p.parts[0] == 'src' and p.suffix == '.py':
        module = '.'.join(p.with_suffix('').relative_to('src').parts)
        importlib.import_module(module.removesuffix('.__init__'))
from unified_hoi.data import read_manifest
import yaml
for name in ('train_behave', 'train_omomo', 'unified'):
    cfg = yaml.safe_load(Path('configs', name + '.yaml').read_text(encoding='utf-8'))
    assert cfg['representation'] == 'parametric_v1'
    for key in ('manifest', 'contact_support', 'contact_codec', 'body_models'):
        p = Path(cfg[key])
        assert p.exists() and not targeted(p), 'Deleted configuration dependency: ' + str(p)
for dataset in ('behave', 'omomo'):
    for row in read_manifest(Path('data/processed', dataset + '_parametric.jsonl')):
        refs = [row[k] for k in ('path', 'text_features_path', 'object_mesh_path') if row.get(k)]
        refs += [v['text_features_path'] for v in row.get('text_variants', []) if v.get('text_features_path')]
        for ref in refs:
            assert Path(ref).is_file() and not targeted(Path(ref)), 'Deleted data dependency: ' + ref
if sys.argv[1] == 'pre':
    for e in r['deletions']:
        p = Path(e['path'])
        if e['deleted'] and not p.exists():
            continue
        files = sorted(f for f in p.rglob('*') if f.is_file()) if p.is_dir() else [p]
        rows = [(f.as_posix(), f.stat().st_size, f.stat().st_mtime_ns) for f in files]
        assert hashlib.sha256(json.dumps(rows).encode()).hexdigest() == e['inventory_sha256'], 'Target changed: ' + str(p)
else:
    assert all(not p.exists() for p in targets), 'Deletion targets remain'
print('Current data, codecs, body assets, raw OMOMO, reviewed source and runtime imports verified.', flush=True)
'@

$python = Join-Path $cleanupRoot '.venv/Scripts/python.exe'
Push-Location -LiteralPath $cleanupRoot
try {
    $report.deletions | Select-Object path, files, bytes, deleted | Format-Table -AutoSize
    Write-Output ('Planned deletion: {0} files, {1:N3} GB. Verifying retained files...' -f $report.planned_files, ($report.planned_bytes / 1000000000))
    $verify | & $python -B - pre
    if ($LASTEXITCODE -ne 0) { throw 'Pre-deletion verification failed.' }
    if (-not $Execute) {
        Write-Output 'Read-only preview passed. Add -Execute to delete the fixed audited targets.'
        return
    }
    $report.status = 'deleting'
    try {
        foreach ($entry in $report.deletions) {
            $target = [IO.Path]::GetFullPath((Join-Path $cleanupRoot $entry.path))
            if ($entry.deleted -and -not (Test-Path -LiteralPath $target)) { continue }
            if (-not $target.StartsWith($cleanupRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw "Target outside project: $target"
            }
            if ($target.Equals($protectedExternal, [StringComparison]::OrdinalIgnoreCase) -or
                $target.StartsWith($protectedExternal + '\', [StringComparison]::OrdinalIgnoreCase)) {
                throw "User-protected external directory: $target"
            }
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
    Write-Output ('Deleted {0} files, {1:N3} GB. Current-model dependencies verified.' -f $report.deleted_files, ($report.deleted_bytes / 1000000000))
} finally {
    Pop-Location
}
