"""Static checks for Git-tracked public files; never executes notebook code."""
from __future__ import annotations

import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = {'.csv', '.tsv', '.xlsx', '.xls', '.parquet', '.jsonl', '.ndjson', '.feather',
           '.arrow', '.db', '.sqlite', '.sqlite3', '.zip', '.gz', '.pdf', '.pptx', '.docx',
           '.dcm', '.dicom', '.nii', '.joblib', '.pkl', '.pickle', '.pt', '.pth',
           '.keras', '.h5', '.npy', '.npz'}
SECRETS = [r'gh[pousr]_[A-Za-z0-9]{30,}', r'github_pat_[A-Za-z0-9_]{30,}',
           r'AKIA[A-Z0-9]{16}', r'sk-(?:proj-)?[A-Za-z0-9_-]{30,}',
           r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
           r'(?i)(?:password|api_key|access_token)\s*=\s*[\'"][^\'"\n]{8,}[\'"]']


def inspect_payload(name: str, payload: bytes) -> list[str]:
    path = Path(name)
    issues = []
    if path.suffix.lower() in BLOCKED or 'PRIVATE' in name:
        issues.append('restricted file type/name')
    directories = {part.lower() for part in path.parts[:-1]}
    if 'data' in directories and path.name != 'README.md':
        issues.append('input data directory')
    if directories & {'outputs', 'results'}:
        issues.append('generated output directory')
    if any(x in path.parts for x in ['.venv', '.analysis_deps', '__pycache__']):
        issues.append('environment/cache')
    if path.name.startswith('.env') and path.name != '.env.example':
        issues.append('environment secret file')
    if len(payload) > 5 * 1024 * 1024:
        issues.append('file larger than 5 MiB')
    if path.suffix.lower() in {'.png', '.jpg'}:
        return issues
    encoding = 'utf-16' if payload.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8'
    try:
        text = payload.decode(encoding)
    except UnicodeError:
        return issues + ['unreviewed binary encoding']
    if any(re.search(p, text) for p in SECRETS):
        issues.append('possible credential (value not printed)')
    absolute_path = (r'(?i)[A-Z]:[/\\](?:Users|Windows)[/\\]|/'
                     + r'content/drive|/' + r'home/[^/\s]+/')
    if re.search(absolute_path, text):
        issues.append('machine-specific absolute path')
    if path.suffix == '.ipynb':
        try:
            nb = json.loads(text)
            assert nb['nbformat'] == 4
            for i, cell in enumerate(nb['cells']):
                if cell['cell_type'] == 'code':
                    if cell.get('outputs') or cell.get('execution_count') is not None:
                        issues.append(f'notebook output/execution count: cell {i}')
                    ast.parse(''.join(cell['source']))
                if cell.get('attachments'):
                    issues.append(f'notebook attachment: cell {i}')
        except (ValueError, KeyError, SyntaxError, AssertionError) as error:
            issues.append(f'invalid notebook: {type(error).__name__}')
    elif path.suffix == '.py':
        try:
            ast.parse(text)
        except SyntaxError:
            issues.append('Python syntax error')
    return issues


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def main() -> int:
    # Index contents (not working-tree files) are the actual next commit candidate.
    issues = []
    entries = git('ls-files', '-s', '-z').decode('utf-8').split('\0')
    count = 0
    for entry in filter(None, entries):
        info, name = entry.split('\t', 1)
        mode, oid, stage = info.split()
        if stage != '0' or mode == '120000':
            issues.append((name, ['unmerged entry or symlink']))
            continue
        found = inspect_payload(name, git('cat-file', 'blob', oid))
        if found:
            issues.append((name, found))
        count += 1
    blobs = set()
    for line in git('rev-list', '--objects', '--all').decode('utf-8').splitlines():
        oid, _, name = line.partition(' ')
        if git('cat-file', '-t', oid).strip() == b'blob':
            blobs.add(oid)
            # Existing history may include files with different publication policy;
            # identify credentials without rewriting any history.
            findings = inspect_payload(name, git('cat-file', 'blob', oid))
            findings = [x for x in findings if 'credential' in x]
            if findings:
                issues.append(('history:' + oid[:12], findings))
    print(json.dumps({'staged_or_tracked_files': count, 'historical_blobs': len(blobs),
                      'issues': issues}, ensure_ascii=False, indent=2))
    return int(bool(issues))


if __name__ == '__main__':
    raise SystemExit(main())
