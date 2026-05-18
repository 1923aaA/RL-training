# 数据格式优化说明

## 修改日期
2026-04-03

## 修改概述
将 `network_performance.json` 数据格式从**全量历史数据**优化为**仅保存最新快照**,由前端负责累积历史数据。

---

## 一、修改内容

### 1. JSON 数据结构变化

#### 旧格式 (已备份为 `network_performance_old_format_backup.json`):
```json
{
  "metadata": { ... },
  "performance_snapshots": [
    { "timestamp": "...", "connections": {...}, "performance": {...} },
    { "timestamp": "...", "connections": {...}, "performance": {...} },
    { "timestamp": "...", "connections": {...}, "performance": {...} }
    // 多个快照,累积增长
  ],
  "networking_info": { ... },
  "controlling_info": { ... }
}
```

#### 新格式 (当前):
```json
{
  "metadata": { ... },
  "latest_snapshot": {
    // 只保留最新一次快照
    "timestamp": "...",
    "connections": {...},
    "performance": {...}
  },
  "networking_info": { ... },
  "controlling_info": { ... }
}
```

---

### 2. 前端代码修改 (`js/index.js`)

#### 修改位置1: 全局常量 (第343行附近)
- 新增: `MAX_HISTORY_DATA_POINTS = 1000` 历史数据上限

#### 修改位置2: `loadBusinessAndNetworkTimingData()` 函数 (第501-670行)
- **兼容性处理**: 支持新旧两种数据格式
  ```javascript
  let latestSnapshot;
  if (data.latest_snapshot) {
      latestSnapshot = data.latest_snapshot;  // 新格式
  } else if (data.performance_snapshots && Array.isArray(data.performance_snapshots)) {
      latestSnapshot = data.performance_snapshots[data.performance_snapshots.length - 1];  // 旧格式兼容
  }
  ```

- **业务表格渲染**: 从循环遍历改为直接处理单个快照
  
- **历史数据累积**: 
  - 原: 遍历所有快照 → 去重 → 累积
  - 新: 检查单个快照时间戳 → 去重 → 累积 → 检查数据上限

- **数据上限管理**: 每个指标超过1000个数据点时自动删除最旧的

---

## 二、优势对比

| 对比项 | 旧方案 | 新方案 |
|-------|--------|--------|
| JSON文件大小 | 随时间增长 (几MB+) | 固定小体积 (~10KB) |
| 网络传输 | 每次传输全部历史 | 每次只传最新数据 |
| 前端处理 | 遍历+去重所有快照 | 只处理单个快照 |
| 内存管理 | 无限制 | 有上限(1000点) |
| 轮询效率 | 随数据增长变慢 | 始终高效 |

---

## 三、后端适配指南

### 生成 JSON 文件的代码需要修改为:

```python
# 示例: Python 后端代码
import json
from datetime import datetime

def generate_performance_json():
    data = {
        "metadata": {
            "test_start_time": test_start_time,
            "description": "Link16标准TDMA网络性能监控数据 - 只保留最新快照",
            "stats_interval": 5
        },
        "latest_snapshot": {
            "timestamp": datetime.now().isoformat(),
            "connections": {
                # ... 最新的连接数据
            },
            "performance": {
                "avg_throughput_mbps": 23.192,
                "avg_latency_ms": 0.03467,
                "avg_packet_delivery_rate": 100.0,
                "avg_jitter_ms": 0.01803
            }
        },
        "networking_info": {
            "average_networking_time": "151",
            "latest_networking_time": "128"
        },
        "controlling_info": {
            "average_controlling_time": "2.1",
            "latest_controlling_time": "2.23"
        }
    }
    
    # 覆盖写入,不追加
    with open('network_performance.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
```

---

## 四、前端工作机制

### 数据流程:

```
后端每5秒:
  覆盖写入 latest_snapshot
  ↓
前端轮询(每5秒):
  下载 latest_snapshot
  ↓
  检查 timestamp 是否已处理
  ↓ (未处理)
  累积到内存数组:
    - accumulatedChartData.throughput
    - accumulatedChartData.latency
    - accumulatedChartData.pdr
    - accumulatedChartData.jitter
  ↓
  检查数组长度 > 1000?
  ↓ (是)
  删除最旧数据点 (array.shift())
  ↓
  渲染ECharts图表
```

### 关键变量:
- `accumulatedChartData`: 内存中的历史数据数组
- `processedTimestamps`: 已处理的时间戳记录(去重)
- `currentSimStartTime`: 当前仿真启动时间(检测新仿真)
- `MAX_HISTORY_DATA_POINTS`: 数据点数量上限(1000)

---

## 五、兼容性说明

前端代码已支持新旧两种格式:
- ✅ 如果 JSON 包含 `latest_snapshot` → 使用新格式
- ✅ 如果 JSON 包含 `performance_snapshots` → 兼容旧格式(取最后一个)

**建议**: 后端尽快切换到新格式以享受性能提升

---

## 六、测试验证

### 测试步骤:
1. 打开浏览器 F12 控制台
2. 观察日志:
   - "首次加载完成，后续将累积新数据"
   - "检测到新仿真启动，清空历史数据" (切换仿真时)
3. 观察图表:
   - 左下角4个折线图应正常显示
   - 数据点随时间增加
4. 检查网络请求:
   - `network_performance.json` 文件大小应维持在 ~10KB

---

## 七、回滚方案

如需回滚到旧格式:
1. 恢复备份文件: 
   ```bash
   Copy-Item "network_performance_old_format_backup.json" "network_performance.json"
   ```
2. 前端代码无需修改(已兼容旧格式)

---

## 八、相关文件

| 文件 | 说明 |
|------|------|
| `network_performance.json` | 新格式数据文件 |
| `network_performance_old_format_backup.json` | 旧格式备份 |
| `js/index.js` | 前端逻辑(已修改) |
| `DATA_FORMAT_UPDATE.md` | 本说明文档 |

---

## 联系方式
如有问题,请检查浏览器控制台日志或联系开发团队。
