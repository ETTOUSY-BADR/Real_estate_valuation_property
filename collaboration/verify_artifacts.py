"""Verify downloaded artifact bytes; --staged also checks Git's staged payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--staged', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'collaboration/artifact_manifest.json').read_text())
    lfs_count = 0
    for entry in manifest['files']:
        path = ROOT / entry['path']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != entry['bytes'] or digest != entry['sha256']:
            raise SystemExit(f"Local artifact mismatch: {entry['path']}")
        if args.staged:
            payload = subprocess.check_output([
                'git', '-c', f'safe.directory={ROOT.as_posix()}', '-C', str(ROOT),
                'show', ':' + entry['path'],
            ])
            if payload.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
                pointer = payload.decode('ascii').splitlines()
                if f"oid sha256:{digest}" not in pointer or f"size {entry['bytes']}" not in pointer:
                    raise SystemExit(f"LFS pointer mismatch: {entry['path']}")
                lfs_count += 1
            elif hashlib.sha256(payload).hexdigest() != digest:
                raise SystemExit(f"Staged artifact mismatch: {entry['path']}")
    print(f"Verified {len(manifest['files'])} artifact checksums.")
    if args.staged:
        print(f"Verified all staged payloads, including {lfs_count} LFS pointers.")


if __name__ == '__main__':
    main()