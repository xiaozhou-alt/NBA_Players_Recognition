# build.spec
block_cipher = None

a = Analysis(
    ['demo.py'],  # 主程序文件
    pathex=[],
    binaries=[],
    datas=[
        ('data', 'data'),             # 球员图像数据
        ('assets', 'assets'),           # 图标和动画资源
        ('output', 'output')            # 模型和映射文件
    ],
    hiddenimports=[
        'tensorflow',
        'tensorflow.keras',
        'PIL',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='NBA-Star-Recognition',  # EXE文件名
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # 不显示控制台窗口
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/nba_logo.ico',  # 应用图标
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='NBA-Star-Recognition',  # 输出文件夹名
)