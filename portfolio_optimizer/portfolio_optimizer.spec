# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Portfolio Optimizer.
Build with: pyinstaller portfolio_optimizer.spec
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, copy_metadata, collect_submodules

block_cipher = None

# Get the directory containing this spec file
spec_dir = Path(SPECPATH)

# Collect all necessary data files
datas = [
    # Main streamlit app
    (str(spec_dir / 'streamlit_app.py'), '.'),
    # MT5 parser module
    (str(spec_dir / 'mt5_parser.py'), '.'),
    # Monte Carlo module
    (str(spec_dir / 'monte_carlo.py'), '.'),
    # Remaining local modules (missing these caused ModuleNotFoundError in the exe)
    (str(spec_dir / 'correlation.py'), '.'),
    (str(spec_dir / 'optimizer.py'), '.'),
    (str(spec_dir / 'generate_report.py'), '.'),
    (str(spec_dir / 'api.py'), '.'),
]

# Add package metadata (fixes "No package metadata was found" errors)
# Using try/except for packages that may not need metadata copying
packages_to_copy_metadata = ['streamlit', 'altair', 'pandas', 'numpy', 'plotly', 'packaging']
for pkg in packages_to_copy_metadata:
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

# Collect streamlit data files
datas += collect_data_files('streamlit')
datas += collect_data_files('altair')
datas += collect_data_files('plotly')

# Hidden imports that PyInstaller might miss
hiddenimports = [
    # Streamlit and its dependencies
    'streamlit',
    'streamlit.web.cli',
    'streamlit.runtime.scriptrunner',
    'streamlit.runtime.caching',
    'streamlit.components.v1',

    # Data processing
    'pandas',
    'numpy',
    'scipy',
    'scipy.stats',
    'scipy.optimize',

    # Plotting
    'plotly',
    'plotly.express',
    'plotly.graph_objects',
    'plotly.subplots',

    # Excel export
    'xlsxwriter',

    # Other dependencies
    'PIL',
    'PIL._imagingtk',
    'altair',
    'pyarrow',
    'validators',
    'gitpython',
    'pydeck',
    'watchdog',
    'tornado',
    'click',
    'toml',
    'packaging',
    # importlib_metadata removed: built into Python 3.12, listing it broke the build
]

# Add all streamlit submodules
hiddenimports += collect_submodules('streamlit')

a = Analysis(
    ['app_launcher.py'],
    pathex=[str(spec_dir)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
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
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='PortfolioOptimizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console for status messages
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one: icon='icon.ico'
)
