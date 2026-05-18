"""
AFSIM <-> Python 的 DDS 桥接层 (MAPPO 版, AFSIM 集成完整版)。

职责:
1. 订阅 AFSIM 发出的:
   - PlatformAdded(1002) / PlatformDeleted(1003) / MoveUpdate(1004)
   - TrackEvent(5001) 传感器 track 事件 [AFSIM 新增]
   - StrikeResult(5002) 打击结果事件 [AFSIM 新增]
2. 把收到的平台消息解析, 自动通过 NodeMapper 注册到 node_id
3. 维护线程安全的 (node_id -> pos/alive/timestamp) 快照
4. 维护线程安全的 track cache (track_id_str -> TrackSnapshot)
5. 维护线程安全的 strike event 列表 (按时间顺序)
6. 接收 MAPPO 的批量控制命令, 打包成 JSON + 'C' 头, 通过 DDS Topic 9001 下发:
   - EntityMoveTo (移动)
   - FireWeapon (开火) [AFSIM 新增]
   - SimRestart (AFSIM 原生硬重启；episode 间不再在线随机撒点)
7. 处理 LLA <-> 局部 ENU (km) 坐标转换

协议格式 (和 C++ 侧 WsfByteStreamSerializer 对齐):
- uint32/uint64: 大端 (network order)
- double       : 主机序 (x86 小端), 因为 C++ 侧 WriteDouble 的 ToNetworkOrder 被注释
- string       : uint32_be length prefix + raw UTF-8 bytes

详细的 payload 字节布局在各自解析函数的 docstring 里。
"""
import json
import logging
import math
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from onpolicy.envs.afsim_kill_chain.core import NodeType
from onpolicy.envs.afsim_kill_chain.node_mapping import NodeMapper, PlatformInfo


# ==================== Topic 常量 ====================
TOPIC_PLATFORM_ADDED   = 1002
TOPIC_PLATFORM_DELETED = 1003
TOPIC_MOVE_UPDATE      = 1004
TOPIC_TRACK_STATE      = 5001  # [AFSIM 新增] TrackEvent (传感器 track 更新)
TOPIC_STRIKE_RESULT    = 5002  # [AFSIM 新增] StrikeResult (武器事件)
TOPIC_MESSAGE_RECEIVED = 5003  # [AFSIM 新增] MessageReceived (comm 栈消息到达, D/I 收到 TrackMessage)
TOPIC_CONTROL_COMMAND  = 9001

CONTROL_MAGIC = ord('C')   # 67, Python -> AFSIM 控制命令的标识字节

# WGS84 地球半径 (小范围足够精度)
EARTH_RADIUS_M = 6378137.0


# ==================== pydds 适配层 ====================
# 同事给的 pydds 包的 API 是:
#   pydds.init(domain_id) -> int (0 = 成功)
#   pydds.subscribe(topic_id, callback)
#   pydds.publish(topic_id, bytes)
#   pydds.shutdown()
try:
    import pydds  # type: ignore
    _PYDDS_AVAILABLE = True
except ImportError:
    pydds = None  # type: ignore
    _PYDDS_AVAILABLE = False


def _dds_init(domain_id: int) -> bool:
    if not _PYDDS_AVAILABLE:
        return False
    result = pydds.init(domain_id)
    return result == 0


def _dds_subscribe(topic_id: int, callback: Callable) -> bool:
    if not _PYDDS_AVAILABLE:
        return False
    try:
        pydds.subscribe(topic_id, callback)
        return True
    except Exception:
        return False


def _dds_unsubscribe(topic_id: int) -> bool:
    if not _PYDDS_AVAILABLE:
        return False
    try:
        pydds.unsubscribe(topic_id)
        return True
    except Exception:
        return False


def _dds_publish(topic_id: int, data: bytes) -> bool:
    if not _PYDDS_AVAILABLE:
        return False
    try:
        pydds.publish(topic_id, data)
        return True
    except Exception:
        return False


def _dds_shutdown():
    if _PYDDS_AVAILABLE:
        try:
            pydds.shutdown()
        except Exception:
            pass


# ==================== 二进制 payload 解析工具 ====================
class _PayloadReader:
    """
    读 AFSIM 发来的二进制 payload, 处理字节序混用的情况。
    """
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def read_uint32_be(self) -> int:
        if self.offset + 4 > len(self.data):
            raise ValueError(f"not enough bytes for uint32 at offset {self.offset}")
        v = struct.unpack_from(">I", self.data, self.offset)[0]
        self.offset += 4
        return v

    def read_uint64_be(self) -> int:
        if self.offset + 8 > len(self.data):
            raise ValueError(f"not enough bytes for uint64 at offset {self.offset}")
        v = struct.unpack_from(">Q", self.data, self.offset)[0]
        self.offset += 8
        return v

    def read_double_le(self) -> float:
        """C++ 侧 WriteDouble 没做字节序转换, x86 是小端"""
        if self.offset + 8 > len(self.data):
            raise ValueError(f"not enough bytes for double at offset {self.offset}")
        v = struct.unpack_from("<d", self.data, self.offset)[0]
        self.offset += 8
        return v

    def read_uint8(self) -> int:
        if self.offset + 1 > len(self.data):
            raise ValueError(f"not enough bytes for uint8 at offset {self.offset}")
        v = self.data[self.offset]
        self.offset += 1
        return v

    def read_string(self) -> str:
        """uint32_be 长度前缀 + UTF-8 bytes"""
        length = self.read_uint32_be()
        if length == 0:
            return ""
        if self.offset + length > len(self.data):
            raise ValueError(f"string length {length} exceeds remaining bytes at offset {self.offset}")
        s = self.data[self.offset:self.offset + length].decode("utf-8", errors="replace")
        self.offset += length
        return s

    def bytes_left(self) -> int:
        return len(self.data) - self.offset


# ==================== 数据结构 ====================
@dataclass
class NodeSnapshot:
    """单个节点的当前状态快照 (从 PlatformAdded / MoveUpdate)"""
    node_id: int
    pos_km: np.ndarray             # 局部 ENU, 单位 km
    alive: bool
    last_update_wall_time: float   # AFSIM 的 wall_time
    last_update_local_time: float  # Python 接收时的本地时间 (time.time())
    heading_rad: float = 0.0
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    speed_mps: float = 0.0


@dataclass
class TrackSnapshot:
    """
    [AFSIM 新增] AFSIM 传感器产生的 track 快照 (topic 5001)。

    一个 track_id_str 唯一标识一条 track；兼容 AFSIM 原生 "platform.local" 和历史桥接层 "platform:local" 格式,
    env 在 SENSOR 分支用 bridge.get_tracks_for_sensor(sensor_node_id) 拿到
    当前这个 sensor 看到的所有 track.
    I 节点发 FireWeapon 时必须用 track_id_str 作为唯一目标标识.
    """
    track_id_str: str                       # "SENSOR_1.5" 或历史 "SENSOR_1:5", 发 FireWeapon/SendTrackMessage 要的关键
    sensor_platform_index: int
    sensor_node_id: int                     # bridge 已转换到 node_id
    sensor_name: str
    sensor_side: str
    msg_type: str                           # "TrackCreated" / "TrackUpdated" / "TrackRemoved" / "TrackUpdate"
    target_platform_index: int
    target_node_id: int                     # -1 如果 target 没在 mapper 里注册
    target_lat_deg: float
    target_lon_deg: float
    target_alt_m: float
    # 转换后的局部 ENU 坐标 (km), 方便 env 用;
    # 如果 bridge 还没建立 LLA 原点, 这里会是 None
    target_pos_km: Optional[np.ndarray]
    target_vel_ned_mps: Tuple[float, float, float]  # (north, east, down)
    range_m: float                          # -1 = invalid
    bearing_deg: float                      # -1 = invalid
    quality: float                          # 0.0 ~ 1.0
    track_age_sec: float
    is_confirmed: bool
    wall_time: float                        # AFSIM sim/wall time
    last_update_local_time: float           # Python time.time()


@dataclass
class StrikeEvent:
    """
    [AFSIM 新增] AFSIM 武器事件 (topic 5002)。

    一次完整的打击生命周期会产生多条事件, 按时间先后:
      WEAPON_FIRED → (WEAPON_HIT | WEAPON_MISS) → WEAPON_TERMINATED
    env 在 _process_strike_events 里按 event_type 和 target_killed 做结算.
    """
    event_type: str                         # "WEAPON_FIRED" / "WEAPON_HIT" / "WEAPON_MISS" / "WEAPON_TERMINATED"
    wall_time: float
    shooter_platform_index: int
    shooter_node_id: int                    # bridge 已转换; -1 如果不在 mapper
    target_platform_index: int              # 0 = 未知
    target_node_id: int                     # -1 如果 0 或不在 mapper
    weapon_name: str
    track_id_str: str                       # 关联到具体 engagement 的关键
    shooter_lla: Tuple[float, float, float]
    target_lla: Tuple[float, float, float]
    miss_distance_m: float                  # -1 = 未知
    hit: bool
    target_killed: bool
    received_local_time: float


@dataclass
class MessageArrivalEvent:
    """
    [AFSIM 新增] AFSIM comm 栈消息到达事件 (topic 5003).

    当 AFSIM 内部的 WsfTrackMessage 通过 comm 组件最终送达某个平台时, C++
    StateObserver::OnMessageReceived 会通过 5003 推这条事件给 Python.
    env 在 _process_message_arrivals 里按 receiver 角色分流:
      - receiver 是 DECISION -> S 送来的情报, 推进 task 到 ASSIGNED-candidate 状态, 让 D 下步决策派 I
      - receiver 是 INTERCEPTOR -> D 派发的指控, 推进 task 到 STRIKING 候选, 让 I 下步决策开火

    注: 中继跳不会出现在这里, C++ 侧已经按 dstAddr == receiver.address 过滤过.
    """
    sim_time: float                         # AFSIM 仿真时间 (秒)
    receiver_platform_index: int
    receiver_node_id: int                   # bridge 已转换; -1 如果不在 mapper
    sender_platform_index: int              # 上一跳发送者; 单跳时 = originator
    sender_node_id: int
    originator_platform_index: int          # 消息最初来源 (S 或 D)
    originator_node_id: int
    msg_class_name: str                     # "WsfTrackMessage"
    track_id_str: str                       # "sensor_1:5" — 供后续 send_fire_weapon 用
    track_local_number: int                 # 5 — 供后续 send_track_message 用
    target_platform_index: int              # track 指向的 target 平台号 (0 = 未知)
    target_node_id: int                     # -1 如果未知或不在 mapper
    target_lat_deg: float
    target_lon_deg: float
    target_alt_m: float
    target_pos_km: Optional[np.ndarray]     # 已转为局部 ENU km; LLA 原点未建时为 None
    quality: float                          # track 质量 0.0~1.0
    received_local_time: float              # bridge 收到这条事件的本地 wall time


# ==================== TrackId 工具 ====================
def _extract_track_local_number(track_id_str: str, default: int = 0) -> int:
    """兼容 AFSIM 原生 owner.local 和历史桥接 owner:local 两种 TrackId 字符串。"""
    if not track_id_str:
        return int(default)
    sep_pos = max(track_id_str.rfind('.'), track_id_str.rfind(':'))
    if sep_pos < 0 or sep_pos + 1 >= len(track_id_str):
        return int(default)
    try:
        return int(track_id_str[sep_pos + 1:])
    except Exception:
        return int(default)


# ==================== Bridge 主体 ====================
class AFSIMBridge:
    """
    AFSIM DDS 桥接层 (完整 OODA 闭环版本)。

    使用流程:
        bridge = AFSIMBridge(cfg, mapper)
        bridge.start()                              # 初始化 DDS, 订阅 1002/1003/1004/5001/5002
        ok, reason = bridge.wait_for_platforms_ready(timeout_sec=15)
        if not ok: raise RuntimeError(reason)

        # ... env.step() 循环里 ...
        # 1. S 节点读 track
        tracks = bridge.get_tracks_for_sensor(sensor_node_id)

        # 2. 下发移动
        bridge.send_move_commands_batch({node_id: target_pos_km, ...})

        # 3. I 节点开火
        bridge.send_fire_weapon(shooter_node_id, track_id_str)

        # 4. sleep 等 AFSIM 推进
        bridge.wait_step(1.0)

        # 5. 读最新状态
        positions = bridge.get_positions()

        # 6. 取打击反馈事件
        events = bridge.pop_strike_events()

        # ... episode 结束 ...
        bridge.soft_reset(randomize=True, seed=episode_seed)   # 发 SimReset + 清内存
        bridge.stop()                               # 关闭 DDS
    """

    def __init__(self, cfg, node_mapper: NodeMapper, mock_mode: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            cfg: afsim_kill_chain.config.Config 实例
            node_mapper: NodeMapper 实例 (和本 bridge 共享)
            mock_mode: 测试用. True 时不真正连 DDS, 所有 publish 记录在内存,
                       subscribe 收不到回调 (外部用 mock_inject_* 注入)
            logger: 可选日志器
        """
        self.cfg = cfg
        self.afsim_cfg = cfg.afsim
        self.mapper = node_mapper
        self.mock_mode = mock_mode
        self.logger = logger or logging.getLogger(__name__)

        # 线程安全锁 (DDS 回调和主线程都会访问共享状态)
        self._lock = threading.RLock()

        # node_id -> NodeSnapshot (PlatformAdded / MoveUpdate 的产物)
        self._snapshots: Dict[int, NodeSnapshot] = {}

        # [AFSIM 新增] track_id_str -> TrackSnapshot (TrackEvent 的产物)
        self._track_cache: Dict[str, TrackSnapshot] = {}

        # [AFSIM 新增] 收到的 strike event 按顺序缓存, env 每 step pop 一次
        self._strike_events: List[StrikeEvent] = []

        # [AFSIM 新增] 收到的 message arrival event 按顺序缓存, env 每 step pop 一次
        # 承载 D 收到 S 的情报 + I 收到 D 的指控 两种场景 (receiver 角色由 env 分流)
        self._message_arrivals: List[MessageArrivalEvent] = []

        # LLA 坐标原点。默认采用 config 中固定 zone 中心，避免每个 episode 由第一个平台自动切换原点。
        self._lla_origin: Optional[Tuple[float, float, float]] = None
        self._init_fixed_lla_origin()

        # DDS 状态
        self._dds_initialized = False
        self._subscribed_topics: set = set()

        # 统计
        self._stats = {
            "platform_added_recv": 0,
            "platform_deleted_recv": 0,
            "move_update_recv": 0,
            "track_event_recv": 0,           # [AFSIM 新增]
            "strike_event_recv": 0,          # [AFSIM 新增]
            "message_arrival_recv": 0,       # [AFSIM 新增] 5003 MESSAGE_RECEIVED 累计条数
            "ctrl_cmd_sent": 0,
            "fire_weapon_sent": 0,           # [AFSIM 新增]
            "send_track_message_sent": 0,    # [AFSIM 新增] 调 bridge.send_track_message 的累计次数
            "sim_reset_sent": 0,             # [AFSIM 新增]
            "parse_errors": 0,
        }

        # Mock 模式缓存
        self._mock_sent_commands: List[dict] = []

        # 平台就绪信号
        self._ready_event = threading.Event()

    def _interaction_log_enabled(self) -> bool:
        return bool(getattr(self.afsim_cfg, "interaction_debug_log", True))

    def _blog(self, text: str):
        if self._interaction_log_enabled():
            self.logger.info(text)

    # ==========================================================
    # 生命周期
    # ==========================================================
    def start(self) -> bool:
        """
        初始化 DDS 并订阅所有必需 topic (1002/1003/1004/5001/5002).
        mock 模式下直接返回 True.
        """
        if self.mock_mode:
            self.logger.info("[Bridge] mock mode, skip DDS init")
            self._dds_initialized = True
            return True

        if not _PYDDS_AVAILABLE:
            self.logger.error("[Bridge] pydds not available, cannot start")
            return False

        ok = _dds_init(self.afsim_cfg.domain_id)
        if not ok:
            self.logger.error(f"[Bridge] pydds.init(domain={self.afsim_cfg.domain_id}) failed")
            return False

        self._dds_initialized = True
        self.logger.info(f"[Bridge] DDS init OK, domain_id={self.afsim_cfg.domain_id}")

        # 订阅 topic (平台状态)
        if not _dds_subscribe(TOPIC_PLATFORM_ADDED, self._on_platform_added):
            self.logger.error(f"[Bridge] failed to subscribe topic {TOPIC_PLATFORM_ADDED}")
            return False
        self._subscribed_topics.add(TOPIC_PLATFORM_ADDED)

        if not _dds_subscribe(TOPIC_PLATFORM_DELETED, self._on_platform_deleted):
            self.logger.warning(f"[Bridge] failed to subscribe topic {TOPIC_PLATFORM_DELETED}")
        else:
            self._subscribed_topics.add(TOPIC_PLATFORM_DELETED)

        if not _dds_subscribe(TOPIC_MOVE_UPDATE, self._on_move_update):
            self.logger.error(f"[Bridge] failed to subscribe topic {TOPIC_MOVE_UPDATE}")
            return False
        self._subscribed_topics.add(TOPIC_MOVE_UPDATE)

        # [AFSIM 新增] 订阅 track / strike
        if not _dds_subscribe(TOPIC_TRACK_STATE, self._on_track_event):
            self.logger.warning(f"[Bridge] failed to subscribe topic {TOPIC_TRACK_STATE} (TRACK_EVENT)")
        else:
            self._subscribed_topics.add(TOPIC_TRACK_STATE)

        if not _dds_subscribe(TOPIC_STRIKE_RESULT, self._on_strike_result):
            self.logger.warning(f"[Bridge] failed to subscribe topic {TOPIC_STRIKE_RESULT} (STRIKE_RESULT)")
        else:
            self._subscribed_topics.add(TOPIC_STRIKE_RESULT)

        # [AFSIM 新增] 订阅 message received (D/I 收到 TrackMessage 的回传)
        if not _dds_subscribe(TOPIC_MESSAGE_RECEIVED, self._on_message_received):
            self.logger.warning(f"[Bridge] failed to subscribe topic {TOPIC_MESSAGE_RECEIVED} (MESSAGE_RECEIVED)")
        else:
            self._subscribed_topics.add(TOPIC_MESSAGE_RECEIVED)

        self.logger.info(f"[Bridge] subscribed topics: {sorted(self._subscribed_topics)}")
        return True

    def stop(self):
        """关闭 DDS"""
        if not self._dds_initialized:
            return

        if not self.mock_mode:
            for topic_id in list(self._subscribed_topics):
                _dds_unsubscribe(topic_id)
            self._subscribed_topics.clear()
            _dds_shutdown()

        self._dds_initialized = False
        self.logger.info("[Bridge] stopped")

    def soft_reset(self, randomize: bool = False, seed: int = 42) -> bool:
        """仅清 Python 本地缓存，不再向 AFSIM 发送 SimReset。

        当前版本的 episode 重置由 AFSIM 原生 SimRestart 完成：
            Python -> {"MsgType":"SimRestart"} -> AFSIM RequestReset()。

        因此这里无论 randomize 参数是什么，都不会生成节点位置、不会发送 SimReset、
        不会改动 AFSIM 平台位置。平台位置完全以 AFSIM 场景重新运行后的状态为准。
        """
        if randomize:
            self.logger.info(
                "[Bridge] soft_reset(randomize=True) ignored: current mode uses SimRestart; no SimReset/random positioning sent"
            )
        self.clear_runtime_state(clear_mapper=True, clear_origin=False)
        self.logger.info("[Bridge] local soft reset done (no AFSIM SimReset)")
        return True

    # ==========================================================
    # 平台就绪等待
    # ==========================================================
    def wait_for_platforms_ready(self, timeout_sec: Optional[float] = None) -> Tuple[bool, str]:
        """
        阻塞等待所有蓝方节点 (S+D+I) 通过 PlatformAdded 注册完毕。
        TARGET 不强制等齐。
        """
        if timeout_sec is None:
            timeout_sec = self.afsim_cfg.reset_timeout_sec

        start_t = time.time()
        while time.time() - start_t < timeout_sec:
            ready, reason = self.mapper.is_mapping_ready()
            if ready:
                self._ready_event.set()
                return True, ""
            time.sleep(0.1)

        ready, reason = self.mapper.is_mapping_ready()
        return False, f"timeout after {timeout_sec}s; {reason}"

    # ==========================================================
    # 控制命令 (Python -> AFSIM)
    # ==========================================================
    def send_move_command(self, node_id: int, target_pos_km: np.ndarray) -> bool:
        """
        给单个节点发移动命令。
        转换 local_km -> LLA, 打包 JSON, publish 到 Topic 9001。
        """
        platform_index = self.mapper.node_to_platform(node_id)
        if platform_index is None:
            self.logger.warning(f"[Bridge] send_move: node_id={node_id} not mapped")
            return False

        if self._lla_origin is None:
            self.logger.warning("[Bridge] send_move: LLA origin not set, cannot convert")
            return False

        lat, lon, alt = self._local_km_to_lla(
            float(target_pos_km[0]), float(target_pos_km[1]), float(target_pos_km[2])
        )

        # 对齐 AFSIM MsgParse.cpp::onEntityMoveTo 的 JSON schema
        cmd = {
            "MsgType": "EntityMoveTo",
            "PlatformId": int(platform_index),
            "Location": [lat, lon, alt],
        }
        return self._publish_control(cmd)

    def send_move_commands_batch(self, commands: Dict[int, np.ndarray]) -> int:
        """批量发送移动命令, 返回成功发送的数量。"""
        success_count = 0
        for node_id, target in commands.items():
            if self.send_move_command(node_id, target):
                success_count += 1
        if bool(getattr(self.afsim_cfg, "motion_debug_log", False)):
            self.logger.info(f"[Bridge-MOVE] EntityMoveTo batch published {success_count}/{len(commands)}")
        return success_count

    def send_fire_weapon(self, shooter_node_id: int, track_id_str: str,
                         target_lat_deg: float = 0.0,
                         target_lon_deg: float = 0.0,
                         target_alt_m: float = 0.0,
                         target_platform_index: int = 0) -> bool:
        """
        [AFSIM 新增] 发 FireWeapon 命令给 AFSIM。

        对齐 C++ MsgParse.cpp::onFireWeapon 的 JSON schema:
            {"MsgType":"FireWeapon", "PlatformId":<int>, "TrackId":<str>,
             "TargetLat":<float>, "TargetLon":<float>, "TargetAlt":<float>,
             "TargetPlatformIndex":<int>}
        其中 TargetLat/Lon/Alt/PlatformIndex 是 fallback 信息,
        当 AFSIM 在 firer/assigner 平台都找不到 RawTrack 时, 用这些信息构造一个临时 track 强制发射.
        这样保证: 即使 AFSIM sensor track 已老化,只要 Python 还知道 target 位置,fire 一定成功.

        Args:
            shooter_node_id: Python 侧的 node_id (I 节点, 通常 9~13)
            track_id_str: 要打击的 track, 来源于 bridge.get_tracks_for_sensor() 的返回
            target_lat_deg: target 最近一次已知位置 (latitude, 度). 0 表示无 fallback.
            target_lon_deg: target 最近一次已知位置 (longitude, 度).
            target_alt_m:   target 最近一次已知位置 (altitude, 米).
            target_platform_index: target 在 AFSIM 里的 platform index (>0 才生效).

        Returns:
            True 表示 DDS publish 成功 (不保证 AFSIM 侧真的开火)
        """
        if not track_id_str:
            self.logger.warning(f"[Bridge] send_fire_weapon: empty track_id_str for shooter={shooter_node_id}")
            return False

        platform_index = self.mapper.node_to_platform(shooter_node_id)
        if platform_index is None:
            self.logger.warning(
                f"[Bridge] send_fire_weapon: shooter node_id={shooter_node_id} not mapped"
            )
            return False

        cmd = {
            "MsgType": "FireWeapon",
            "PlatformId": int(platform_index),
            "TrackId": str(track_id_str),
            # [FIX bug #fire-with-fallback-target] 附带 target 的最近已知位置, C++ 找不到 track 时用它重建.
            "TargetLat": float(target_lat_deg),
            "TargetLon": float(target_lon_deg),
            "TargetAlt": float(target_alt_m),
            "TargetPlatformIndex": int(target_platform_index),
        }
        ok = self._publish_control(cmd)
        if ok:
            self._stats["fire_weapon_sent"] += 1
            self.logger.info(
                f"[Bridge] FireWeapon sent: shooter_node={shooter_node_id} "
                f"platform={platform_index} track='{track_id_str}' "
                f"fallback_target=({target_lat_deg:.6f}, {target_lon_deg:.6f}, {target_alt_m:.1f}m, "
                f"platform_idx={target_platform_index})"
            )
        else:
            self.logger.warning(
                f"[Bridge] FireWeapon publish failed: shooter_node={shooter_node_id} track='{track_id_str}'"
            )
        return ok

    def send_track_message(self, src_node_id: int, next_hop_node_id: int,
                           track_id_str: str = "",
                           track_local_number: int = 0,
                           target_lat_deg: float = 0.0,
                           target_lon_deg: float = 0.0,
                           target_alt_m: float = 0.0,
                           target_platform_index: int = 0) -> bool:
        """
        [AFSIM 新增] 让 AFSIM src 平台通过 comm 栈发一条原生 WsfTrackMessage 给 nextHop 平台。

        关键原则：
        - Python 不修改 WsfTrackMessage 格式，也不把 task_id/msg_id 塞进 WsfTrackMessage；
        - Python 只下发转发指令，AFSIM 端用 TrackId/TrackManager 查询真实 WsfTrack；
        - AFSIM comm 真正送达后，5003 MESSAGE_RECEIVED 再通知 Python 更新 D/I buffer。

        推荐 JSON schema:
            {"MsgType":"SendTrackMessage",
             "SrcPlatformId":<int>, "NextHopPlatformId":<int>,
             "TrackId":<str>, "TrackLocalNumber":<int>,
             "TargetLat":<float>, "TargetLon":<float>, "TargetAlt":<float>,
             "TargetPlatformIndex":<int>}

        TrackId 兼容两种格式：
            AFSIM 原生: "owner.local"
            历史桥接: "owner:local"

        Returns:
            True  — DDS publish 成功；不保证 AFSIM comm 必然送达，送达以 5003 为准。
            False — 参数校验失败 / 未映射 / DDS publish 失败。
        """
        if not track_id_str and track_local_number <= 0:
            self.logger.warning(
                f"[Bridge] send_track_message: empty TrackId and invalid local_number={track_local_number} "
                f"src_node={src_node_id} next_hop_node={next_hop_node_id}"
            )
            return False

        if track_local_number <= 0:
            track_local_number = _extract_track_local_number(track_id_str, 0)

        src_platform = self.mapper.node_to_platform(src_node_id)
        if src_platform is None:
            self.logger.warning(
                f"[Bridge] send_track_message: src node_id={src_node_id} not mapped"
            )
            return False
        next_hop_platform = self.mapper.node_to_platform(next_hop_node_id)
        if next_hop_platform is None:
            self.logger.warning(
                f"[Bridge] send_track_message: next_hop node_id={next_hop_node_id} not mapped"
            )
            return False

        cmd = {
            "MsgType": "SendTrackMessage",
            "SrcPlatformId": int(src_platform),
            "NextHopPlatformId": int(next_hop_platform),
            # C++ 端优先使用 TrackId，通过 TrackManager 的 FindTrack/FindCorrelatedTrack/FindRawTrack 查询真实 WsfTrack。
            "TrackId": str(track_id_str or ""),
            # 保留 TrackLocalNumber 是为了兼容旧 C++ 和日志诊断。
            "TrackLocalNumber": int(track_local_number),
            "TargetLat": float(target_lat_deg),
            "TargetLon": float(target_lon_deg),
            "TargetAlt": float(target_alt_m),
            "TargetPlatformIndex": int(target_platform_index),
        }
        ok = self._publish_control(cmd)
        if ok:
            self._stats["send_track_message_sent"] += 1
            self.logger.info(
                f"[Bridge] SendTrackMessage sent: src_node={src_node_id} (plat={src_platform}) "
                f"→ next_hop_node={next_hop_node_id} (plat={next_hop_platform}) "
                f"track='{track_id_str}' local={track_local_number} "
                f"target_platform={target_platform_index}"
            )
        else:
            self.logger.warning(
                f"[Bridge] SendTrackMessage publish failed: "
                f"src_node={src_node_id} → next_hop_node={next_hop_node_id} "
                f"track='{track_id_str}' local={track_local_number}"
            )
        return ok

    def send_raw_command(self, cmd_dict: dict) -> bool:
        """发送任意自定义 JSON 命令"""
        return self._publish_control(cmd_dict)

    def wait_step(self, duration_sec: Optional[float] = None):
        """sleep 一个 step, 让 AFSIM 推进物理"""
        if duration_sec is None:
            duration_sec = self.afsim_cfg.step_wait_sec
        time.sleep(duration_sec)

    def _publish_control(self, cmd_dict: dict) -> bool:
        """内部: 打包 + 发送控制命令"""
        json_str = json.dumps(cmd_dict, separators=(",", ":"))
        payload = struct.pack(">I", CONTROL_MAGIC) + json_str.encode("utf-8")

        if self.mock_mode:
            self._mock_sent_commands.append(cmd_dict)
            self._stats["ctrl_cmd_sent"] += 1
            return True

        ok = _dds_publish(TOPIC_CONTROL_COMMAND, payload)
        if ok:
            self._stats["ctrl_cmd_sent"] += 1
        if self.afsim_cfg.dds_debug_log:
            self.logger.debug(f"[Bridge] publish control: {json_str[:120]}")
        elif self._interaction_log_enabled() and cmd_dict.get("MsgType") != "EntityMoveTo":
            self.logger.info(f"[Bridge-9001] publish {cmd_dict.get('MsgType')} ok={ok} payload={json_str[:220]}")
        return ok

    # ==========================================================
    # 快照读取 (AFSIM -> Python)
    # ==========================================================
    def get_positions(self) -> Dict[int, np.ndarray]:
        """所有已知节点的当前位置 (km). 没有更新过的节点不在返回值里。"""
        with self._lock:
            return {nid: snap.pos_km.copy() for nid, snap in self._snapshots.items()}

    def get_alive_map(self) -> Dict[int, bool]:
        with self._lock:
            return {nid: snap.alive for nid, snap in self._snapshots.items()}

    def get_snapshot(self, node_id: int) -> Optional[NodeSnapshot]:
        with self._lock:
            s = self._snapshots.get(node_id)
            if s is None:
                return None
            return NodeSnapshot(
                node_id=s.node_id, pos_km=s.pos_km.copy(), alive=s.alive,
                last_update_wall_time=s.last_update_wall_time,
                last_update_local_time=s.last_update_local_time,
                heading_rad=s.heading_rad, pitch_rad=s.pitch_rad, roll_rad=s.roll_rad,
                speed_mps=s.speed_mps,
            )

    def get_all_snapshots(self) -> Dict[int, NodeSnapshot]:
        with self._lock:
            result = {}
            for nid, s in self._snapshots.items():
                result[nid] = NodeSnapshot(
                    node_id=s.node_id, pos_km=s.pos_km.copy(), alive=s.alive,
                    last_update_wall_time=s.last_update_wall_time,
                    last_update_local_time=s.last_update_local_time,
                    heading_rad=s.heading_rad, pitch_rad=s.pitch_rad, roll_rad=s.roll_rad,
                    speed_mps=s.speed_mps,
                )
            return result

    def get_stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    # ==========================================================
    # [AFSIM 新增] Track 查询接口
    # ==========================================================
    def get_tracks_for_sensor(self, sensor_node_id: int,
                               include_stale: bool = False) -> List[TrackSnapshot]:
        """
        返回当前某个 sensor node 正在观测的所有 track (按 range_m 升序).

        过滤规则:
        - sensor_node_id 必须匹配
        - msg_type 为 "TrackRemoved" 的条目不返回 (在回调里已删除)
        - include_stale=False 时 (默认), 丢弃 local_time 超过 track_stale_sec 的条目

        Args:
            sensor_node_id: bridge 已转换过的 node_id (0..n_sensor-1)
            include_stale: 是否包括过期 track (默认不包)

        Returns:
            List[TrackSnapshot] 拷贝, 调用方可安全修改. 按 range_m 升序排列
            (range_m < 0 的视为 inf 排到最后).
        """
        now = time.time()
        stale_threshold = float(self.afsim_cfg.track_stale_sec)

        with self._lock:
            results: List[TrackSnapshot] = []
            for t in self._track_cache.values():
                if t.sensor_node_id != sensor_node_id:
                    continue
                if not include_stale:
                    age_local = now - t.last_update_local_time
                    if age_local > stale_threshold:
                        continue
                # 复制一份, 避免外部修改污染缓存
                results.append(TrackSnapshot(
                    track_id_str=t.track_id_str,
                    sensor_platform_index=t.sensor_platform_index,
                    sensor_node_id=t.sensor_node_id,
                    sensor_name=t.sensor_name,
                    sensor_side=t.sensor_side,
                    msg_type=t.msg_type,
                    target_platform_index=t.target_platform_index,
                    target_node_id=t.target_node_id,
                    target_lat_deg=t.target_lat_deg,
                    target_lon_deg=t.target_lon_deg,
                    target_alt_m=t.target_alt_m,
                    target_pos_km=(None if t.target_pos_km is None else t.target_pos_km.copy()),
                    target_vel_ned_mps=t.target_vel_ned_mps,
                    range_m=t.range_m,
                    bearing_deg=t.bearing_deg,
                    quality=t.quality,
                    track_age_sec=t.track_age_sec,
                    is_confirmed=t.is_confirmed,
                    wall_time=t.wall_time,
                    last_update_local_time=t.last_update_local_time,
                ))

        # 按 range_m 升序, 无效 range (-1) 放最后
        def _sort_key(tr: TrackSnapshot) -> float:
            return tr.range_m if tr.range_m >= 0 else 1e18
        results.sort(key=_sort_key)
        return results

    def get_track_by_id(self, track_id_str: str) -> Optional[TrackSnapshot]:
        """按 track_id_str 查最新 track snapshot (不过滤 stale)"""
        with self._lock:
            t = self._track_cache.get(track_id_str)
            if t is None:
                return None
            return TrackSnapshot(
                track_id_str=t.track_id_str,
                sensor_platform_index=t.sensor_platform_index,
                sensor_node_id=t.sensor_node_id,
                sensor_name=t.sensor_name,
                sensor_side=t.sensor_side,
                msg_type=t.msg_type,
                target_platform_index=t.target_platform_index,
                target_node_id=t.target_node_id,
                target_lat_deg=t.target_lat_deg,
                target_lon_deg=t.target_lon_deg,
                target_alt_m=t.target_alt_m,
                target_pos_km=(None if t.target_pos_km is None else t.target_pos_km.copy()),
                target_vel_ned_mps=t.target_vel_ned_mps,
                range_m=t.range_m,
                bearing_deg=t.bearing_deg,
                quality=t.quality,
                track_age_sec=t.track_age_sec,
                is_confirmed=t.is_confirmed,
                wall_time=t.wall_time,
                last_update_local_time=t.last_update_local_time,
            )

    def clear_stale_tracks(self) -> int:
        """
        清理超过 track_stale_sec 没更新的 track. 返回清理数量.
        env 可以定期调用 (或者直接依赖 get_tracks_for_sensor 的过滤).
        """
        now = time.time()
        stale_threshold = float(self.afsim_cfg.track_stale_sec)
        with self._lock:
            to_del = [k for k, v in self._track_cache.items()
                      if (now - v.last_update_local_time) > stale_threshold]
            for k in to_del:
                del self._track_cache[k]
            return len(to_del)

    # ==========================================================
    # [AFSIM 新增] Strike event 查询接口
    # ==========================================================
    def pop_strike_events(self) -> List[StrikeEvent]:
        """
        原子取走所有累积的打击事件并清空内部列表.
        env 应该在 step() 里每步调用一次.

        Returns:
            List[StrikeEvent], 按接收顺序 (不一定严格按 wall_time, 但通常接近)
        """
        with self._lock:
            events = self._strike_events
            self._strike_events = []
        return events

    def peek_strike_events(self) -> List[StrikeEvent]:
        """只看不取 (用于调试/日志)"""
        with self._lock:
            return list(self._strike_events)

    # ==========================================================
    # [AFSIM 新增] Message arrival event 查询接口
    # ==========================================================
    def pop_message_arrivals(self) -> List[MessageArrivalEvent]:
        """
        原子取走所有累积的 comm 消息到达事件并清空内部列表.
        env 应该在 step() 里每步调用一次 (放在 _process_strike_events 之后、
        action 处理之前, 让 D/I 能在当前 step 就看到刚到达的消息).

        Returns:
            List[MessageArrivalEvent], 按接收顺序.
        """
        with self._lock:
            evs = self._message_arrivals
            self._message_arrivals = []
        return evs

    def peek_message_arrivals(self) -> List[MessageArrivalEvent]:
        """只看不取 (用于调试/日志)"""
        with self._lock:
            return list(self._message_arrivals)

    # ==========================================================
    # 坐标转换 (LLA <-> 局部 ENU km)
    # ==========================================================

    def _init_fixed_lla_origin(self):
        """按配置初始化固定 LLA 原点。

        这个原点只用于 Python 侧 LLA <-> 局部 ENU(km) 坐标转换，
        不会改变 AFSIM 场景中任何平台的位置。
        """
        if getattr(self.afsim_cfg, "use_fixed_lla_origin", False):
            self._lla_origin = (
                float(getattr(self.afsim_cfg, "fixed_origin_lat", 0.0)),
                float(getattr(self.afsim_cfg, "fixed_origin_lon", 0.0)),
                float(getattr(self.afsim_cfg, "fixed_origin_alt", 0.0)),
            )
            self.logger.info(
                f"[Bridge] fixed LLA origin set: "
                f"({self._lla_origin[0]:.6f}, {self._lla_origin[1]:.6f}, {self._lla_origin[2]:.2f})"
            )

    def clear_runtime_state(self, clear_mapper: bool = True, clear_origin: bool = False):
        """清 Python 侧运行时缓存，不向 AFSIM 发送任何 reset/randomize 指令。

        用于 SimRestart 前后丢弃上一回合残留状态。
        clear_origin 默认 False，因为训练区域不变，局部坐标原点不应每回合切换。
        """
        with self._lock:
            self._snapshots.clear()
            self._track_cache.clear()
            self._strike_events.clear()
            self._message_arrivals.clear()
            self._ready_event.clear()
            if clear_origin:
                self._lla_origin = None
                self._init_fixed_lla_origin()
        if clear_mapper:
            self.mapper.clear()
        self.logger.info("[Bridge] runtime state cleared (no SimReset sent)")

    def set_lla_origin(self, lat_deg: float, lon_deg: float, alt_m: float):
        """显式设置 LLA 原点。当前训练推荐固定为 zone 中心，不随 episode 切换。"""
        with self._lock:
            self._lla_origin = (lat_deg, lon_deg, alt_m)
        self.logger.info(f"[Bridge] LLA origin set: ({lat_deg:.6f}, {lon_deg:.6f}, {alt_m:.2f})")

    def _lla_to_local_km(self, lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
        """
        LLA -> 局部 ENU (km). 对 < 20km 范围, 用简单平面投影近似已经足够 (误差 < 1m).
        """
        if self._lla_origin is None:
            # 正常情况下 __init__ 已按配置固定原点。只有用户显式关闭固定原点时才会走到这里。
            self._lla_origin = (lat_deg, lon_deg, alt_m)
            self.logger.info(
                f"[Bridge] auto-set LLA origin to first platform because fixed origin is disabled: "
                f"({lat_deg:.6f}, {lon_deg:.6f}, {alt_m:.2f})"
            )

        lat0, lon0, alt0 = self._lla_origin
        lat0_rad = math.radians(lat0)
        dy_m = math.radians(lat_deg - lat0) * EARTH_RADIUS_M
        dx_m = math.radians(lon_deg - lon0) * EARTH_RADIUS_M * math.cos(lat0_rad)
        dz_m = alt_m - alt0
        return np.array([dx_m / 1000.0, dy_m / 1000.0, dz_m / 1000.0], dtype=np.float32)

    def _local_km_to_lla(self, x_km: float, y_km: float, z_km: float) -> Tuple[float, float, float]:
        """局部 ENU (km) -> LLA"""
        if self._lla_origin is None:
            raise RuntimeError("LLA origin not set (no platform registered yet)")
        lat0, lon0, alt0 = self._lla_origin
        lat0_rad = math.radians(lat0)
        dlat_deg = math.degrees((y_km * 1000.0) / EARTH_RADIUS_M)
        dlon_deg = math.degrees((x_km * 1000.0) / (EARTH_RADIUS_M * math.cos(lat0_rad)))
        return lat0 + dlat_deg, lon0 + dlon_deg, alt0 + z_km * 1000.0

    # ==========================================================
    # DDS 回调 (在 DDS 线程执行) -- 平台事件
    # ==========================================================
    # ==========================================================
    # DDS 回调 (在 DDS 线程执行) -- 平台事件
    # ==========================================================
    def _on_platform_added(self, topic_id: int, payload: bytes):
        """Topic 1002"""
        try:
            info_dict = self._parse_platform_payload(payload, expected_id=TOPIC_PLATFORM_ADDED)
            if info_dict is None:
                return

            pos_km = self._lla_to_local_km(info_dict["lat"], info_dict["lon"], info_dict["alt"])

            pinfo = PlatformInfo(
                platform_index=info_dict["platform_index"],
                name=info_dict.get("name", f"plat_{info_dict['platform_index']}"),
                side=info_dict.get("side", ""),
                x_km=float(pos_km[0]),
                y_km=float(pos_km[1]),
                z_km=float(pos_km[2]),
            )

            # --------------------------------------------------
            # 关键修改：
            # 在 register_platform() 之前判断该平台是否已经注册过。
            #
            # 如果没有注册过：
            #   这条 1002 日志解释为真正的 PlatformAdded。
            #
            # 如果已经注册过：
            #   这条 1002 日志解释为 AFSIM 周期性平台快照刷新，
            #   不再误打印成“平台重复添加”。
            # --------------------------------------------------
            was_registered = pinfo.platform_index in self.mapper._platform_to_node

            node_id = self.mapper.register_platform(pinfo)

            with self._lock:
                self._stats["platform_added_recv"] += 1

                if node_id is not None:
                    self._snapshots[node_id] = NodeSnapshot(
                        node_id=node_id,
                        pos_km=pos_km,
                        alive=True,
                        last_update_wall_time=info_dict.get("wall_time", 0.0),
                        last_update_local_time=time.time(),
                        heading_rad=info_dict.get("heading", 0.0),
                        pitch_rad=info_dict.get("pitch", 0.0),
                        roll_rad=info_dict.get("roll", 0.0),
                        speed_mps=info_dict.get("speed", 0.0),
                    )

            if node_id is not None:
                if not was_registered:
                    self.logger.info(
                        f"[Bridge-1002] PlatformAdded: platform={pinfo.platform_index} "
                        f"side='{pinfo.side}' -> node_id={node_id} pos={pos_km}"
                    )
                else:
                    self.logger.info(
                        f"[Bridge-1002-SNAPSHOT] PlatformSnapshot refresh: "
                        f"platform={pinfo.platform_index} "
                        f"side='{pinfo.side}' node_id={node_id} pos={pos_km}"
                    )
            else:
                self.logger.debug(
                    f"[Bridge-1002] PlatformAdded ignored: platform={pinfo.platform_index} "
                    f"side='{pinfo.side}' not classified"
                )

        except Exception as e:
            with self._lock:
                self._stats["parse_errors"] += 1
            self.logger.exception(f"[Bridge-1002] _on_platform_added parse error: {e}")

    def _on_platform_deleted(self, topic_id: int, payload: bytes):
        """
        Topic 1003.
        [修复] 之前只读了 identifier + platform_index, 现在按 C++ 实际发的格式读全:
            u32_be identifier(1003) + u32_be platform_index + f64_le wall_time
            + [string name] + [string side]
        没有 LLA 信息 (删除事件 C++ 侧没写位置).
        """
        try:
            reader = _PayloadReader(payload)
            ident = reader.read_uint32_be()
            if ident != TOPIC_PLATFORM_DELETED:
                self.logger.warning(f"[Bridge] PlatformDeleted expected 1003, got {ident}")
                return
            platform_index = reader.read_uint32_be()

            # 选读 wall_time (C++ SendPlatformEventViaDDS 对 PlatformDeleted 也会写 wall_time)
            wall_time = 0.0
            if reader.bytes_left() >= 8:
                try:
                    wall_time = reader.read_double_le()
                except Exception:
                    wall_time = 0.0

            # 选读 name/side (如果 has_name_side)
            has_name_side = getattr(self.afsim_cfg, "platform_msg_has_name_side", True)
            if has_name_side and reader.bytes_left() >= 4:
                try:
                    _name = reader.read_string()
                    _side = reader.read_string()
                except Exception:
                    pass

            node_id = self.mapper.unregister_platform(platform_index)

            with self._lock:
                self._stats["platform_deleted_recv"] += 1
                if node_id is not None and node_id in self._snapshots:
                    # 保留 snapshot 条目, 仅标记死亡, 方便 env 侧 alive_map 识别
                    snap = self._snapshots[node_id]
                    self._snapshots[node_id] = NodeSnapshot(
                        node_id=snap.node_id, pos_km=snap.pos_km, alive=False,
                        last_update_wall_time=wall_time,
                        last_update_local_time=time.time(),
                        heading_rad=snap.heading_rad, pitch_rad=snap.pitch_rad,
                        roll_rad=snap.roll_rad, speed_mps=snap.speed_mps,
                    )

            self.logger.info(
                f"[Bridge] PlatformDeleted: platform={platform_index} "
                f"node_id={node_id} wall_time={wall_time:.2f}"
            )
        except Exception as e:
            with self._lock:
                self._stats["parse_errors"] += 1
            self.logger.exception(f"[Bridge] _on_platform_deleted error: {e}")

    def _on_move_update(self, topic_id: int, payload: bytes):
        """Topic 1004. 若平台未注册 (PlatformAdded 丢失), 按 MoveUpdate 兜底注册."""
        try:
            info_dict = self._parse_platform_payload(payload, expected_id=TOPIC_MOVE_UPDATE)
            if info_dict is None:
                return

            platform_index = info_dict["platform_index"]
            node_id = self.mapper.platform_to_node(platform_index)

            # ===== 兜底注册: PlatformAdded 可能在 Python 订阅前就发完了 =====
            if node_id is None:
                try:
                    pos_km = self._lla_to_local_km(
                        info_dict["lat"], info_dict["lon"], info_dict["alt"]
                    )
                except Exception as e:
                    self.logger.warning(
                        f"[Bridge] auto-register skipped, lla->local failed: {e}"
                    )
                    return

                pinfo = PlatformInfo(
                    platform_index=platform_index,
                    name=info_dict.get("name", f"plat_{platform_index}"),
                    side=info_dict.get("side", ""),
                    x_km=float(pos_km[0]),
                    y_km=float(pos_km[1]),
                    z_km=float(pos_km[2]),
                )
                node_id = self.mapper.register_platform(pinfo)
                if node_id is None:
                    # 角色不匹配 (explicit 列表以外) 或该角色已满, 忽略
                    return

                with self._lock:
                    self._stats["platform_added_recv"] += 1
                self.logger.info(
                    f"[Bridge] auto-register from MoveUpdate: "
                    f"platform={platform_index} name='{pinfo.name}' side='{pinfo.side}' "
                    f"pos=({pos_km[0]:.2f},{pos_km[1]:.2f},{pos_km[2]:.2f}) -> node_id={node_id}"
                )

                # 检查是否所有蓝方就绪
                ready, _ = self.mapper.is_mapping_ready()
                if ready:
                    self._ready_event.set()

            # ===== 更新快照 =====
            pos_km = self._lla_to_local_km(
                info_dict["lat"], info_dict["lon"], info_dict["alt"]
            )
            with self._lock:
                self._stats["move_update_recv"] += 1
                prev_alive = self._snapshots[node_id].alive if node_id in self._snapshots else True
                self._snapshots[node_id] = NodeSnapshot(
                    node_id=node_id,
                    pos_km=pos_km,
                    alive=prev_alive,
                    last_update_wall_time=info_dict.get("wall_time", 0.0),
                    last_update_local_time=time.time(),
                    heading_rad=info_dict.get("heading", 0.0),
                    pitch_rad=info_dict.get("pitch", 0.0),
                    roll_rad=info_dict.get("roll", 0.0),
                    speed_mps=info_dict.get("speed", 0.0),
                )
        except Exception as e:
            with self._lock:
                self._stats["parse_errors"] += 1
            self.logger.exception(f"[Bridge] _on_move_update error: {e}")

    # ==========================================================
    # [AFSIM 新增] DDS 回调 -- Track / Strike 事件
    # ==========================================================
    def _on_track_event(self, topic_id: int, payload: bytes):
        """Topic 5001 TRACK_EVENT. C++ 侧 SendTrackEventViaDDS 发的 payload."""
        try:
            td = self._parse_track_event_payload(payload)
            if td is None:
                return

            # sensor 的 platform_index -> node_id
            sensor_node_id = self.mapper.platform_to_node(td["sensor_platform_index"])
            if sensor_node_id is None:
                # sensor 还没注册, skip (PlatformAdded 可能还在路上)
                with self._lock:
                    self._stats["parse_errors"] += 0  # 不算解析错误, 只是暂时跳过
                self.logger.debug(
                    f"[Bridge] track event for unregistered sensor platform "
                    f"{td['sensor_platform_index']}, skip"
                )
                return

            # target 的 platform_index -> node_id (target 没注册也继续, node_id 填 -1)
            target_node_id = -1
            if td["target_platform_index"] > 0:
                mapped = self.mapper.platform_to_node(td["target_platform_index"])
                if mapped is not None:
                    target_node_id = mapped

            # target LLA -> 局部 ENU km (如果 LLA 原点已建立)
            target_pos_km = None
            try:
                if self._lla_origin is not None:
                    target_pos_km = self._lla_to_local_km(
                        td["target_lat"], td["target_lon"], td["target_alt"]
                    )
            except Exception:
                target_pos_km = None

            snap = TrackSnapshot(
                track_id_str=td["track_id_str"],
                sensor_platform_index=td["sensor_platform_index"],
                sensor_node_id=sensor_node_id,
                sensor_name=td["sensor_name"],
                sensor_side=td["sensor_side"],
                msg_type=td["msg_type"],
                target_platform_index=td["target_platform_index"],
                target_node_id=target_node_id,
                target_lat_deg=td["target_lat"],
                target_lon_deg=td["target_lon"],
                target_alt_m=td["target_alt"],
                target_pos_km=target_pos_km,
                target_vel_ned_mps=(td["tgt_vel_n"], td["tgt_vel_e"], td["tgt_vel_d"]),
                range_m=td["range_m"],
                bearing_deg=td["bearing_deg"],
                quality=td["quality"],
                track_age_sec=td["track_age"],
                is_confirmed=bool(td["is_confirmed"]),
                wall_time=td["wall_time"],
                last_update_local_time=time.time(),
            )

            with self._lock:
                self._stats["track_event_recv"] += 1
                if td["msg_type"] == "TrackRemoved":
                    self._track_cache.pop(td["track_id_str"], None)
                else:
                    self._track_cache[td["track_id_str"]] = snap

            if self._interaction_log_enabled() or self.afsim_cfg.dds_debug_log:
                n = max(1, int(getattr(self.afsim_cfg, "track_event_log_every_n", 1)))
                if (self._stats.get("track_event_recv", 0) % n) == 0 or td["msg_type"] == "TrackRemoved":
                    self.logger.info(
                        f"[Bridge-5001] TrackEvent {td['msg_type']} track='{td['track_id_str']}' "
                        f"sensor_node={sensor_node_id} sensor_platform={td['sensor_platform_index']} "
                        f"target_node={target_node_id} target_platform={td['target_platform_index']} "
                        f"range={td['range_m']:.0f}m bearing={td['bearing_deg']:.1f} q={td['quality']:.2f} "
                        f"confirmed={bool(td['is_confirmed'])} cache_size={len(self._track_cache)}"
                    )
        except Exception as e:
            with self._lock:
                self._stats["parse_errors"] += 1
            self.logger.exception(f"[Bridge] _on_track_event error: {e}")

    def _on_strike_result(self, topic_id: int, payload: bytes):
        """Topic 5002 STRIKE_RESULT."""
        try:
            ev = self._parse_strike_result_payload(payload)
            if ev is None:
                return

            # shooter 的 platform_index -> node_id
            shooter_node_id = -1
            if ev["shooter_platform_index"] > 0:
                mapped = self.mapper.platform_to_node(ev["shooter_platform_index"])
                if mapped is not None:
                    shooter_node_id = mapped

            # target 同理
            target_node_id = -1
            if ev["target_platform_index"] > 0:
                mapped = self.mapper.platform_to_node(ev["target_platform_index"])
                if mapped is not None:
                    target_node_id = mapped

            strike_ev = StrikeEvent(
                event_type=ev["event_type"],
                wall_time=ev["wall_time"],
                shooter_platform_index=ev["shooter_platform_index"],
                shooter_node_id=shooter_node_id,
                target_platform_index=ev["target_platform_index"],
                target_node_id=target_node_id,
                weapon_name=ev["weapon_name"],
                track_id_str=ev["track_id_str"],
                shooter_lla=(ev["shooter_lat"], ev["shooter_lon"], ev["shooter_alt"]),
                target_lla=(ev["target_lat"], ev["target_lon"], ev["target_alt"]),
                miss_distance_m=ev["miss_distance"],
                hit=bool(ev["hit"]),
                target_killed=bool(ev["target_killed"]),
                received_local_time=time.time(),
            )

            with self._lock:
                self._stats["strike_event_recv"] += 1
                self._strike_events.append(strike_ev)

            self.logger.info(
                f"[Bridge] StrikeEvent: {ev['event_type']} shooter_node={shooter_node_id} "
                f"target_node={target_node_id} weapon='{ev['weapon_name']}' "
                f"track='{ev['track_id_str']}' hit={ev['hit']} killed={ev['target_killed']} "
                f"miss={ev['miss_distance']:.0f}m"
            )
        except Exception as e:
            with self._lock:
                self._stats["parse_errors"] += 1
            self.logger.exception(f"[Bridge] _on_strike_result error: {e}")

    # ==========================================================
    # [AFSIM 新增] DDS 回调 -- Comm 消息到达事件 (5003 MESSAGE_RECEIVED)
    # ==========================================================
    def _on_message_received(self, topic_id: int, payload: bytes):
        """
        Topic 5003 MESSAGE_RECEIVED. 由 C++ StateObserver::OnMessageReceived 触发,
        表示某个平台通过 comm 栈最终收到了一条 WsfTrackMessage.
        中继跳不会触发 (C++ 侧按 dstAddr == receiver.address 过滤过).
        """
        try:
            md = self._parse_message_received_payload(payload)
            if md is None:
                return

            # receiver platform_index -> node_id
            receiver_node_id = -1
            mapped = self.mapper.platform_to_node(md["receiver_platform_index"])
            if mapped is not None:
                receiver_node_id = mapped

            # sender platform_index -> node_id
            sender_node_id = -1
            if md["sender_platform_index"] > 0:
                mapped = self.mapper.platform_to_node(md["sender_platform_index"])
                if mapped is not None:
                    sender_node_id = mapped

            # originator platform_index -> node_id
            originator_node_id = -1
            if md["originator_platform_index"] > 0:
                mapped = self.mapper.platform_to_node(md["originator_platform_index"])
                if mapped is not None:
                    originator_node_id = mapped

            # target platform_index -> node_id
            target_node_id = -1
            if md["target_platform_index"] > 0:
                mapped = self.mapper.platform_to_node(md["target_platform_index"])
                if mapped is not None:
                    target_node_id = mapped

            # target LLA -> 局部 ENU km (如果 LLA 原点已建立)
            target_pos_km = None
            try:
                if self._lla_origin is not None:
                    target_pos_km = self._lla_to_local_km(
                        md["target_lat"], md["target_lon"], md["target_alt"]
                    )
            except Exception:
                target_pos_km = None

            ev = MessageArrivalEvent(
                sim_time=md["sim_time"],
                receiver_platform_index=md["receiver_platform_index"],
                receiver_node_id=receiver_node_id,
                sender_platform_index=md["sender_platform_index"],
                sender_node_id=sender_node_id,
                originator_platform_index=md["originator_platform_index"],
                originator_node_id=originator_node_id,
                msg_class_name=md["msg_class_name"],
                track_id_str=md["track_id_str"],
                track_local_number=md["track_local_number"],
                target_platform_index=md["target_platform_index"],
                target_node_id=target_node_id,
                target_lat_deg=md["target_lat"],
                target_lon_deg=md["target_lon"],
                target_alt_m=md["target_alt"],
                target_pos_km=target_pos_km,
                quality=md["quality"],
                received_local_time=time.time(),
            )

            with self._lock:
                self._stats["message_arrival_recv"] += 1
                self._message_arrivals.append(ev)

            if self._interaction_log_enabled() or self.afsim_cfg.dds_debug_log:
                self.logger.info(
                    f"[Bridge-5003] MessageArrival receiver_node={receiver_node_id} "
                    f"receiver_platform={md['receiver_platform_index']} sender_node={sender_node_id} "
                    f"sender_platform={md['sender_platform_index']} originator_node={originator_node_id} "
                    f"track='{md['track_id_str']}' local={md['track_local_number']} "
                    f"target_node={target_node_id} target_platform={md['target_platform_index']} q={md['quality']:.2f} "
                    f"arrival_queue={len(self._message_arrivals)}"
                )
        except Exception as e:
            with self._lock:
                self._stats["parse_errors"] += 1
            self.logger.exception(f"[Bridge] _on_message_received error: {e}")

    # ==========================================================
    # 协议解析
    # ==========================================================
    def _parse_platform_payload(self, payload: bytes, expected_id: int) -> Optional[dict]:
        """
        解析 PlatformAdded (1002) / MoveUpdate (1004) 的 payload.
        两者格式一致 (PlatformDeleted 后面没有 lat/lon... 因此单独处理).
        """
        reader = _PayloadReader(payload)

        ident = reader.read_uint32_be()
        if ident != expected_id:
            self.logger.warning(f"[Bridge] expected identifier {expected_id}, got {ident}")
            return None

        platform_index = reader.read_uint32_be()
        wall_time = reader.read_double_le()

        # name/side 容错
        name = ""
        side = ""
        has_name_side = getattr(self.afsim_cfg, "platform_msg_has_name_side", True)

        if has_name_side:
            try:
                name = reader.read_string()
                side = reader.read_string()
            except Exception:
                self.logger.warning(
                    "[Bridge] failed to parse name/side; "
                    "if AFSIM C++ 侧没应用'改动A', 请设 cfg.afsim.platform_msg_has_name_side=False"
                )
                return None

        try:
            lat = reader.read_double_le()
            lon = reader.read_double_le()
            alt = reader.read_double_le()
            heading = reader.read_double_le()
            pitch = reader.read_double_le()
            roll = reader.read_double_le()
            speed = reader.read_double_le()
        except Exception as e:
            self.logger.warning(f"[Bridge] failed to parse position/attitude: {e}")
            return None

        return {
            "platform_index": platform_index,
            "wall_time": wall_time,
            "name": name,
            "side": side,
            "lat": lat,
            "lon": lon,
            "alt": alt,
            "heading": heading,
            "pitch": pitch,
            "roll": roll,
            "speed": speed,
        }

    def _parse_track_event_payload(self, payload: bytes) -> Optional[dict]:
        """
        解析 Topic 5001 TRACK_EVENT payload.
        C++ 侧格式 (StateObserver.cpp::SendTrackEventViaDDS):

          u32_be identifier = 5001
          u32_be sensorPlatformId
          f64_le wallTime
          str    sensorName
          str    sensorSide
          str    msgType              ("TrackCreated"/"TrackUpdated"/"TrackRemoved"/...)
          str    trackIdStr           ("sensor_1:5")
          u32_be targetPlatformId
          f64_le targetLat, targetLon, targetAlt
          f64_le targetVelN, targetVelE, targetVelD
          f64_le range                 (m, -1 = invalid)
          f64_le bearingDeg            (度, -1 = invalid)
          f64_le trackQuality          (0.0 ~ 1.0)
          f64_le trackAge              (秒)
          u8     isConfirmed           (quality > 0.5 时为 1)
        """
        reader = _PayloadReader(payload)
        ident = reader.read_uint32_be()
        if ident != TOPIC_TRACK_STATE:
            self.logger.warning(f"[Bridge] track event expected id 5001, got {ident}")
            return None

        try:
            sensor_platform_index = reader.read_uint32_be()
            wall_time = reader.read_double_le()
            sensor_name = reader.read_string()
            sensor_side = reader.read_string()
            msg_type = reader.read_string()
            track_id_str = reader.read_string()
            target_platform_index = reader.read_uint32_be()
            target_lat = reader.read_double_le()
            target_lon = reader.read_double_le()
            target_alt = reader.read_double_le()
            tgt_vel_n = reader.read_double_le()
            tgt_vel_e = reader.read_double_le()
            tgt_vel_d = reader.read_double_le()
            range_m = reader.read_double_le()
            bearing_deg = reader.read_double_le()
            quality = reader.read_double_le()
            track_age = reader.read_double_le()
            is_confirmed = reader.read_uint8()
        except Exception as e:
            self.logger.warning(f"[Bridge] track event payload parse failed: {e}")
            return None

        return {
            "sensor_platform_index": sensor_platform_index,
            "wall_time": wall_time,
            "sensor_name": sensor_name,
            "sensor_side": sensor_side,
            "msg_type": msg_type,
            "track_id_str": track_id_str,
            "target_platform_index": target_platform_index,
            "target_lat": target_lat,
            "target_lon": target_lon,
            "target_alt": target_alt,
            "tgt_vel_n": tgt_vel_n,
            "tgt_vel_e": tgt_vel_e,
            "tgt_vel_d": tgt_vel_d,
            "range_m": range_m,
            "bearing_deg": bearing_deg,
            "quality": quality,
            "track_age": track_age,
            "is_confirmed": is_confirmed,
        }

    def _parse_strike_result_payload(self, payload: bytes) -> Optional[dict]:
        """
        解析 Topic 5002 STRIKE_RESULT payload.
        C++ 侧格式 (StateObserver.cpp::SendStrikeResultViaDDS):

          u32_be identifier = 5002
          f64_le wallTime
          str    eventType     ("WEAPON_FIRED"/"WEAPON_HIT"/"WEAPON_MISS"/"WEAPON_TERMINATED")
          u32_be shooterPlatformId
          u32_be targetPlatformId    (0 = 未知)
          str    weaponName
          str    trackIdStr
          f64_le shooterLat, shooterLon, shooterAlt
          f64_le targetLat, targetLon, targetAlt
          f64_le missDistance  (m, -1 = 未知)
          u8     hit           (1=命中, 0=未命中)
          u8     targetKilled  (1=目标已摧毁, 0=未摧毁)
        """
        reader = _PayloadReader(payload)
        ident = reader.read_uint32_be()
        if ident != TOPIC_STRIKE_RESULT:
            self.logger.warning(f"[Bridge] strike result expected id 5002, got {ident}")
            return None

        try:
            wall_time = reader.read_double_le()
            event_type = reader.read_string()
            shooter_platform_index = reader.read_uint32_be()
            target_platform_index = reader.read_uint32_be()
            weapon_name = reader.read_string()
            track_id_str = reader.read_string()
            shooter_lat = reader.read_double_le()
            shooter_lon = reader.read_double_le()
            shooter_alt = reader.read_double_le()
            target_lat = reader.read_double_le()
            target_lon = reader.read_double_le()
            target_alt = reader.read_double_le()
            miss_distance = reader.read_double_le()
            hit = reader.read_uint8()
            target_killed = reader.read_uint8()
        except Exception as e:
            self.logger.warning(f"[Bridge] strike result payload parse failed: {e}")
            return None

        return {
            "wall_time": wall_time,
            "event_type": event_type,
            "shooter_platform_index": shooter_platform_index,
            "target_platform_index": target_platform_index,
            "weapon_name": weapon_name,
            "track_id_str": track_id_str,
            "shooter_lat": shooter_lat,
            "shooter_lon": shooter_lon,
            "shooter_alt": shooter_alt,
            "target_lat": target_lat,
            "target_lon": target_lon,
            "target_alt": target_alt,
            "miss_distance": miss_distance,
            "hit": hit,
            "target_killed": target_killed,
        }

    def _parse_message_received_payload(self, payload: bytes) -> Optional[dict]:
        """
        解析 Topic 5003 MESSAGE_RECEIVED payload.
        C++ 侧格式 (StateObserver.cpp::SendMessageReceivedViaDDS):

          u32_be identifier = 5003
          f64_le simTime
          u32_be receiver_platform_id     (最终接收平台, D 或 I)
          u32_be sender_platform_id       (上一跳发送平台; 单跳时 = originator)
          u32_be originator_platform_id   (消息最初来源, S 或 D)
          str    msg_script_class_name    ("WsfTrackMessage")
          str    track_id_str             ("sensor_1:5")
          u32_be track_local_number       (5)
          u32_be target_platform_index    (track 指向的 target 平台号, 0 = 未知)
          f64_le target_lat, target_lon, target_alt
          f64_le track_quality            (0.0 ~ 1.0)
        """
        reader = _PayloadReader(payload)
        ident = reader.read_uint32_be()
        if ident != TOPIC_MESSAGE_RECEIVED:
            self.logger.warning(f"[Bridge] message received expected id 5003, got {ident}")
            return None

        try:
            sim_time = reader.read_double_le()
            receiver_platform_index = reader.read_uint32_be()
            sender_platform_index = reader.read_uint32_be()
            originator_platform_index = reader.read_uint32_be()
            msg_class_name = reader.read_string()
            track_id_str = reader.read_string()
            track_local_number = reader.read_uint32_be()
            target_platform_index = reader.read_uint32_be()
            target_lat = reader.read_double_le()
            target_lon = reader.read_double_le()
            target_alt = reader.read_double_le()
            quality = reader.read_double_le()
        except Exception as e:
            self.logger.warning(f"[Bridge] message received payload parse failed: {e}")
            return None

        return {
            "sim_time": sim_time,
            "receiver_platform_index": receiver_platform_index,
            "sender_platform_index": sender_platform_index,
            "originator_platform_index": originator_platform_index,
            "msg_class_name": msg_class_name,
            "track_id_str": track_id_str,
            "track_local_number": int(track_local_number),
            "target_platform_index": target_platform_index,
            "target_lat": target_lat,
            "target_lon": target_lon,
            "target_alt": target_alt,
            "quality": quality,
        }

    # ==========================================================
    # Mock 测试专用接口
    # ==========================================================
    def mock_inject_platform_added(self, platform_index: int, name: str, side: str,
                                    lat_deg: float, lon_deg: float, alt_m: float,
                                    heading: float = 0.0, speed: float = 0.0):
        """测试用: 伪造一条 PlatformAdded 消息喂给 bridge"""
        payload = self._mock_pack_platform_payload(
            TOPIC_PLATFORM_ADDED, platform_index, name, side,
            lat_deg, lon_deg, alt_m, heading=heading, speed=speed,
        )
        self._on_platform_added(TOPIC_PLATFORM_ADDED, payload)

    def mock_inject_move_update(self, platform_index: int,
                                 lat_deg: float, lon_deg: float, alt_m: float,
                                 heading: float = 0.0, speed: float = 0.0,
                                 name: str = "", side: str = ""):
        """测试用: 伪造一条 MoveUpdate 消息"""
        payload = self._mock_pack_platform_payload(
            TOPIC_MOVE_UPDATE, platform_index, name=name, side=side,
            lat_deg=lat_deg, lon_deg=lon_deg, alt_m=alt_m,
            heading=heading, speed=speed,
        )
        self._on_move_update(TOPIC_MOVE_UPDATE, payload)

    def mock_inject_platform_deleted(self, platform_index: int,
                                     wall_time: float = 0.0,
                                     name: str = "", side: str = ""):
        """测试用: 伪造一条 PlatformDeleted 消息"""
        buf = struct.pack(">II", TOPIC_PLATFORM_DELETED, platform_index)
        buf += struct.pack("<d", wall_time)
        has_name_side = getattr(self.afsim_cfg, "platform_msg_has_name_side", True)
        if has_name_side:
            nb = name.encode("utf-8")
            sb = side.encode("utf-8")
            buf += struct.pack(">I", len(nb)) + nb
            buf += struct.pack(">I", len(sb)) + sb
        self._on_platform_deleted(TOPIC_PLATFORM_DELETED, buf)

    def mock_inject_track_event(self, sensor_platform_index: int, sensor_name: str,
                                 sensor_side: str, msg_type: str, track_id_str: str,
                                 target_platform_index: int,
                                 target_lat: float, target_lon: float, target_alt: float,
                                 quality: float = 0.8,
                                 range_m: float = -1.0, bearing_deg: float = -1.0,
                                 tgt_vel_ned: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                 track_age: float = 0.0,
                                 is_confirmed: Optional[bool] = None,
                                 wall_time: float = 0.0):
        """[AFSIM 新增] 测试用: 伪造一条 TrackEvent 消息"""
        if is_confirmed is None:
            is_confirmed = quality > 0.5

        buf = struct.pack(">I", TOPIC_TRACK_STATE)                    # identifier
        buf += struct.pack(">I", int(sensor_platform_index))
        buf += struct.pack("<d", float(wall_time))
        for s in (sensor_name, sensor_side, msg_type, track_id_str):
            sb = s.encode("utf-8")
            buf += struct.pack(">I", len(sb)) + sb
        buf += struct.pack(">I", int(target_platform_index))
        buf += struct.pack("<d", float(target_lat))
        buf += struct.pack("<d", float(target_lon))
        buf += struct.pack("<d", float(target_alt))
        buf += struct.pack("<d", float(tgt_vel_ned[0]))
        buf += struct.pack("<d", float(tgt_vel_ned[1]))
        buf += struct.pack("<d", float(tgt_vel_ned[2]))
        buf += struct.pack("<d", float(range_m))
        buf += struct.pack("<d", float(bearing_deg))
        buf += struct.pack("<d", float(quality))
        buf += struct.pack("<d", float(track_age))
        buf += struct.pack("B", 1 if is_confirmed else 0)

        self._on_track_event(TOPIC_TRACK_STATE, buf)

    def mock_inject_strike_result(self, event_type: str,
                                   shooter_platform_index: int,
                                   target_platform_index: int,
                                   weapon_name: str,
                                   track_id_str: str,
                                   shooter_lla: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                   target_lla: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                   miss_distance: float = -1.0,
                                   hit: bool = False,
                                   target_killed: bool = False,
                                   wall_time: float = 0.0):
        """[AFSIM 新增] 测试用: 伪造一条 StrikeResult 消息"""
        buf = struct.pack(">I", TOPIC_STRIKE_RESULT)                  # identifier
        buf += struct.pack("<d", float(wall_time))

        et = event_type.encode("utf-8")
        buf += struct.pack(">I", len(et)) + et

        buf += struct.pack(">I", int(shooter_platform_index))
        buf += struct.pack(">I", int(target_platform_index))

        wn = weapon_name.encode("utf-8")
        buf += struct.pack(">I", len(wn)) + wn

        tid = track_id_str.encode("utf-8")
        buf += struct.pack(">I", len(tid)) + tid

        for v in shooter_lla:
            buf += struct.pack("<d", float(v))
        for v in target_lla:
            buf += struct.pack("<d", float(v))

        buf += struct.pack("<d", float(miss_distance))
        buf += struct.pack("B", 1 if hit else 0)
        buf += struct.pack("B", 1 if target_killed else 0)

        self._on_strike_result(TOPIC_STRIKE_RESULT, buf)

    def mock_inject_message_received(self,
                                      receiver_platform_index: int,
                                      sender_platform_index: int,
                                      originator_platform_index: int,
                                      track_id_str: str,
                                      track_local_number: int,
                                      target_platform_index: int,
                                      target_lat: float = 0.0,
                                      target_lon: float = 0.0,
                                      target_alt: float = 0.0,
                                      quality: float = 0.8,
                                      sim_time: float = 0.0,
                                      msg_class_name: str = "WsfTrackMessage"):
        """
        [AFSIM 新增] 测试用: 伪造一条 MessageReceived (5003) 消息.
        在 mock 模式下模拟 "S→D 情报到达 D" 或 "D→I 指控到达 I".
        """
        buf = struct.pack(">I", TOPIC_MESSAGE_RECEIVED)          # identifier
        buf += struct.pack("<d", float(sim_time))                # sim_time
        buf += struct.pack(">I", int(receiver_platform_index))   # receiver
        buf += struct.pack(">I", int(sender_platform_index))     # sender (上一跳)
        buf += struct.pack(">I", int(originator_platform_index)) # originator (最初来源)

        # str msg_class_name
        mb = msg_class_name.encode("utf-8")
        buf += struct.pack(">I", len(mb)) + mb
        # str track_id_str
        tb = track_id_str.encode("utf-8")
        buf += struct.pack(">I", len(tb)) + tb
        # u32 track_local_number
        buf += struct.pack(">I", int(track_local_number))
        # u32 target_platform_index
        buf += struct.pack(">I", int(target_platform_index))
        # target lat/lon/alt
        buf += struct.pack("<d", float(target_lat))
        buf += struct.pack("<d", float(target_lon))
        buf += struct.pack("<d", float(target_alt))
        # quality
        buf += struct.pack("<d", float(quality))

        self._on_message_received(TOPIC_MESSAGE_RECEIVED, buf)

    def mock_sent_commands(self) -> List[dict]:
        """测试用: 获取所有已发送的控制命令"""
        return list(self._mock_sent_commands)

    def _mock_pack_platform_payload(self, topic_id: int, platform_index: int,
                                     name: str, side: str,
                                     lat_deg: float, lon_deg: float, alt_m: float,
                                     heading: float = 0.0, pitch: float = 0.0,
                                     roll: float = 0.0, speed: float = 0.0,
                                     wall_time: float = 0.0) -> bytes:
        """测试用: 按真实 C++ 格式打包 payload"""
        buf = b""
        buf += struct.pack(">I", topic_id)                    # identifier (big-endian)
        buf += struct.pack(">I", platform_index)              # platform_index (big-endian)
        buf += struct.pack("<d", wall_time)                   # wall_time (little-endian)

        has_name_side = getattr(self.afsim_cfg, "platform_msg_has_name_side", True)
        if has_name_side:
            name_bytes = name.encode("utf-8")
            side_bytes = side.encode("utf-8")
            buf += struct.pack(">I", len(name_bytes)) + name_bytes
            buf += struct.pack(">I", len(side_bytes)) + side_bytes

        buf += struct.pack("<d", lat_deg)
        buf += struct.pack("<d", lon_deg)
        buf += struct.pack("<d", alt_m)
        buf += struct.pack("<d", heading)
        buf += struct.pack("<d", pitch)
        buf += struct.pack("<d", roll)
        buf += struct.pack("<d", speed)
        return buf