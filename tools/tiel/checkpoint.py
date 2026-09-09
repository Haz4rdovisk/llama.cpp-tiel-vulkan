#!/usr/bin/env python3
"""Local runtime checkpoints. Never overwrite source/builds or stop servers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    for name, expected in manifest['sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise RuntimeError('Checkpoint mismatch: ' + name)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['save', 'verify', 'start'])
    parser.add_argument('directory', type=Path)
    parser.add_argument('--build', type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--source-commit')
    args = parser.parse_args()
    root = args.directory.resolve()
    if args.action == 'save':
        if not args.build or not args.profile or not args.source_commit:
            parser.error('save requires --build, --profile and --source-commit')
        build = args.build.resolve()
        if root == build or root.is_relative_to(build) or build.is_relative_to(root):
            parser.error('checkpoint and build must be separate directories')
        profile = json.loads(args.profile.read_text())
        root.mkdir(parents=True, exist_ok=False)
        shutil.copytree(build / 'bin', root / 'bin', symlinks=False)
        shutil.copy2(args.profile, root / 'profile.json')
        shutil.copy2(__file__, root / 'checkpoint.py')
        if (build / 'CMakeCache.txt').exists():
            shutil.copy2(build / 'CMakeCache.txt', root / 'CMakeCache.txt')
        executable = root / 'bin' / Path(profile['cmd'][0]).name
        # Absolute build RUNPATH must never silently select the current build.
        env = dict(os.environ, LD_LIBRARY_PATH=str(root / 'bin'))
        deps = subprocess.check_output(['ldd', str(executable)], env=env, text=True)
        for line in deps.splitlines():
            if any(n in line for n in ['libllama', 'libggml', 'libmtmd']):
                if str(root / 'bin') not in line:
                    raise RuntimeError('Dependency escaped checkpoint: ' + line)
        (root / 'dependencies.txt').write_text(deps)
        manifest = {'source_commit': args.source_commit,
                    'model_included': False, 'system_libraries_included': False,
                    'sha256': {str(p.relative_to(root)): digest(p)
                               for p in root.rglob('*') if p.is_file()}}
        (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        verify(root)
        print('SAVED_AND_VERIFIED', root)
        return
    manifest = verify(root)
    if args.action == 'verify':
        print('VERIFIED', manifest['source_commit'], len(manifest['sha256']), 'files')
        return
    profile = json.loads((root / 'profile.json').read_text())
    # Refuse any existing llama-server, not only the checkpoint's port.
    probe = subprocess.run(['pgrep', '-x', 'llama-server'], capture_output=True)
    if probe.returncode != 1:
        raise RuntimeError('Stop the confirmed DEV process first; no server is stopped automatically')
    cmd = list(profile['cmd'])
    cmd[0] = str(root / 'bin' / Path(cmd[0]).name)
    host = cmd[cmd.index('--host') + 1]
    port = int(cmd[cmd.index('--port') + 1])
    if port == 8080:
        raise RuntimeError('Production port is forbidden')
    with socket.socket() as sock:
        sock.bind((host, port))
    env = runtime_environment(os.environ, profile['env'], root)
    print('Starting saved runtime in foreground. Restore host frequency policy separately.', flush=True)
    os.execve(cmd[0], cmd, env)


def runtime_environment(inherited, selected, root):
    env = {k: v for k, v in inherited.items()
           if not k.startswith(('GGML_VK_', 'LLAMA_MOE_CACHE_', 'LLAMA_SERVER_', 'LLAMA_TIEL_', 'LD_'))}
    env.pop('RADV_PERFTEST', None)
    env.update(selected)
    env['LD_LIBRARY_PATH'] = str(root / 'bin')
    return env


if __name__ == '__main__':
    main()
