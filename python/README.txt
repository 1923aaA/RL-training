├── python/
│   ├── setup.py                  # 构建脚本
│   ├── pydds_binding.cpp     # Pybind11 C++ 绑定代码
│   ├── pydds/
│   │   ├── __init__.py           # 包初始化
│   │   ├── pydds_config.py       # 配置路径 & DLL 动态加载
│   │   ├── pydds_bindings.cp313-win_amd64.pyd        #构建生成
├── DDSCOMM_SDK/                  # 第三方 DDS SDK
│   ├── include/                  # DDSCOMM.h 等头文件
│   └── lib/                      # DDSCOMM.lib 等依赖 DLL & LIB


准备好setup.py   ，__init__.py  ， pydds_binding.cpp，pydds_config.py，已经需要的依赖库。
python setup.py build_ext --inplace
生成pydds_bindings.cp313-win_amd64.pyd即可使用，
在其他py文件内  
import pydds
pydds.init(DOMAIN_ID)；#初始化
pydds.publish(TOPIC_ID, data)；#发布消息
pydds.subscribe(TOPIC_ID, on_message_received)；#订阅消息


一般来说一个python版本就需要一个pydds_bindings.cp313-win_amd64.pyd，所以换版本之后就需要重新生成。因此我准备了build_all_pydds.bat文件，但是还没测试它的功能，以后再说。
