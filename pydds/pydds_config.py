# pydds_config.py

import os

# ========== Step 1: 检查环境变量中的 SDK_ROOT ==========
SDK_ROOT = os.environ.get("SDK_ROOT")

# 如果 SDK_ROOT 未设置或路径不存在，则使用默认值
DEFAULT_SDK_ROOT = r"D:\DDSCOMM_SDK"

if not SDK_ROOT or not os.path.exists(SDK_ROOT):
    print(f"[PyDDS] ⚠️ 环境变量 SDK_ROOT 未设置或路径不存在，使用默认路径: {DEFAULT_SDK_ROOT}")
    SDK_ROOT = DEFAULT_SDK_ROOT
else:
    print(f"[PyDDS] ✅ 使用环境变量 SDK_ROOT: {SDK_ROOT}")

# ========== Step 2: 子路径拼接 ==========
INCLUDE_DIR = os.path.join(SDK_ROOT, "include")
LIB_DIR = os.path.join(SDK_ROOT, "lib")
ACE_LIB_DIR = os.path.join(LIB_DIR, "ACE_lib")
DDS_LIB_DIR = os.path.join(LIB_DIR, "dds_lib")


# 可选的 DLL 搜索路径（用于 os.add_dll_directory）
DLL_SEARCH_PATHS = [
    LIB_DIR,
    ACE_LIB_DIR,
    DDS_LIB_DIR,
]
