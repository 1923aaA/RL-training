from setuptools import setup, Extension
import pybind11
import os

#  读取 pydds_config.py 中的路径变量
from pydds import pydds_config as config

# 校验路径
if not os.path.exists(config.SDK_ROOT):
    raise FileNotFoundError(f"SDK root not found: {config.SDK_ROOT}")
if not os.path.exists(config.INCLUDE_DIR):
    raise FileNotFoundError(f"Include directory not found: {config.INCLUDE_DIR}")

print(f"[PyDDS] SDK: {config.SDK_ROOT}")
print(f"[PyDDS] INCLUDE: {config.INCLUDE_DIR}")
print(f"[PyDDS] LIB: {config.LIB_DIR}")

# ==================== 编译模块 ====================
ext_modules = [
    Extension(
        name="pydds.pydds_bindings",# 这里是 pydds/pydds_bindings.so|.pyd
        sources=["pydds_binding.cpp"],
        include_dirs=[
            pybind11.get_include(),
            config.INCLUDE_DIR,
        ],
        library_dirs=[
            config.LIB_DIR,
            config.ACE_LIB_DIR,
            config.DDS_LIB_DIR,
        ],
        libraries=[
            "DDSCOMM",
            "OpenDDS_Dcps",
            "OpenDDS_Rtps",
            "OpenDDS_Rtps_Udp",
            "TAO_PortableServer",
            "TAO_AnyTypeCode",
            "TAO_ValueType",
            "TAO",
            "ACE",
            "ws2_32",
            "wsock32",
            "advapi32",
        ],
        extra_compile_args=[
            "/std:c++14",
            "/EHsc",
            "/MD",
            "/W3",
            "/DWIN32",
            "/D_WINDOWS",
        ],
        extra_link_args=["/MACHINE:X64"],
        language="c++",
    )
]
# ==================== 安装配置 ====================
setup(
    name="pydds",
    version="1.0.0",
    author="AeroZeng",
    author_email="zengzenghz@nuaa.edu.cn",
    description="Python bindings for DDSCOMM - OpenDDS Communication Library",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    install_requires=[
        "pybind11>=2.10.0",
    ],
    python_requires=">=3.7",
    zip_safe=False,
    classifiers=[
        # 开发状态：
        #   1 - Planning       : 计划中
        #   2 - Pre-Alpha      : 预发布
        #   3 - Alpha          : Alpha版
        #   4 - Beta           : Beta版（功能完整但可能有bug）
        #   5 - Production/Stable : 稳定版
        #   6 - Mature         : 成熟
        #   7 - Inactive       : 不再维护
        "Development Status :: 4 - Beta",
        # 目标受众：
        #   Developers         : 开发者
        #   End Users/Desktop  : 终端用户
        #   Science/Research   : 科研人员
        "Intended Audience :: Developers",
        # 主题分类：这个包属于什么领域
        #   Software Development :: Libraries : 软件开发库
        #   Communications                    : 通信
        #   System :: Networking              : 网络
        "Topic :: Software Development :: Libraries",
        "Programming Language :: C++",
        # 编程语言：标注使用的编程语言
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
