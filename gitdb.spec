# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for GitDB all-in-one binary."""

import sys
from pathlib import Path

block_cipher = None

a = Analysis(
    ['gitdb/cli.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'gitdb',
        'gitdb.core',
        'gitdb.cli',
        'gitdb.server',
        'gitdb.objects',
        'gitdb.delta',
        'gitdb.documents',
        'gitdb.schema',
        'gitdb.hooks',
        'gitdb.types',
        'gitdb.working_tree',
        'gitdb.structured',
        'gitdb.indexes',
        'gitdb.snapshots',
        'gitdb.watches',
        'gitdb.ambient',
        'gitdb.remote',
        'gitdb.pullrequest',
        'gitdb.backup',
        'gitdb.ingest',
        'gitdb.cloud_ingest',
        'gitdb.storage',
        'gitdb.encryption',
        'gitdb.streaming',
        'gitdb.fhe',
        'gitdb.embed',
        'gitdb.distributed',
        'gitdb.crush',
        'gitdb.grpc_service',
        'torch',
        'zstandard',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'IPython', 'notebook', 'pytest'],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='gitdb',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
