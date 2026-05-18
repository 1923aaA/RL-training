# pydds/__init__.py

import os
import sys

try:
    from . import pydds_config
except ImportError:
    pydds_config = None

def _load_dll_paths():
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        for path in getattr(pydds_config, "DLL_SEARCH_PATHS", []):
            if os.path.exists(path):
                try:
                    os.add_dll_directory(path)
                except Exception as e:
                    print(f"[PyDDS] Failed to add DLL path: {path} - {e}")

_load_dll_paths()

# ✅ 构建时跳过绑定导入
if not (hasattr(sys, "argv") and any("build" in arg for arg in sys.argv)):
    try:
        from .pydds_bindings import *  # 只在运行时加载 .pyd
    except ImportError:
        print("[PyDDS] Warning: pydds_bindings not available (probably building)")
