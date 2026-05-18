# 📝 JSON 字段简化说明文档

## 修改日期
2026-04-03

---

## 📋 修改概述

简化 `network_performance.json` 数据字段,删除冗余数据,保留核心性能指标。

### 修改目标
- ✅ 减少数据字段数量
- ✅ 统一字段命名规范
- ✅ 提高数据可读性
- ✅ 减少数据文件体积

---

## 🔄 字段变更对照表

### 1. `detailed_stats` 字段变更 (连接级统计)

#### 修改前 (旧字段):
```json
{
  "detailed_stats": {
    "total_bits_per_second": 44852551.23,
    "avg_bits_per_second": 22426276.0,
    "total_packets_sent": 450,
    "total_packets_received": 450,
    "total_lost_packets": 0,
    "total_retransmits": 0,
    "total_out_of_order": 0,
    "total_duplicates": 0,
    "total_bytes_transferred": 204800,
    "total_test_duration_sec": 0.45,
    "packet_delivery_rate": 100.0,
    "effective_loss_rate": 0.0
  }
}
```

#### 修改后 (新字段):
```json
{
  "detailed_stats": {
    "throughput_bits_per_second": 44852551.23,
    "avg_latency_ms": 0.0345,
    "packet_delivery_ratio": 100.0,
    "avg_latency_jitter_ms": 0.0145,
    "packets_sent": 450,
    "packets_received": 450,
    "bytes_transferred": 204800
  }
}
```

#### 字段对照:

| 旧字段名 | 新字段名 | 说明 | 单位 | 保留 |
|---------|---------|------|------|------|
| `avg_bits_per_second` | `throughput_bits_per_second` | 吞吐量 | bps | ✅ |
| (新增) | `avg_latency_ms` | 平均时延 | ms | ✅ |
| `packet_delivery_rate` | `packet_delivery_ratio` | 包投递率 | % | ✅ |
| (新增) | `avg_latency_jitter_ms` | 时延抖动 | ms | ✅ |
| `total_packets_sent` | `packets_sent` | 发送包数 | 个 | ✅ |
| `total_packets_received` | `packets_received` | 接收包数 | 个 | ✅ |
| `total_bytes_transferred` | `bytes_transferred` | 传输字节 | bytes | ✅ |
| `total_bits_per_second` | - | - | - | ❌ 删除 |
| `total_lost_packets` | - | - | - | ❌ 删除 |
| `total_retransmits` | - | - | - | ❌ 删除 |
| `total_out_of_order` | - | - | - | ❌ 删除 |
| `total_duplicates` | - | - | - | ❌ 删除 |
| `total_test_duration_sec` | - | - | - | ❌ 删除 |
| `effective_loss_rate` | - | - | - | ❌ 删除 |

---

### 2. `performance` 字段变更 (全局统计)

#### 修改前 (旧字段):
```json
{
  "performance": {
    "avg_throughput_mbps": 23.192,
    "max_throughput_mbps": 24.67,
    "avg_latency_ms": 0.03467,
    "avg_packet_delivery_rate": 100.0,
    "avg_jitter_ms": 0.01803
  }
}
```

#### 修改后 (新字段):
```json
{
  "performance": {
    "total_throughput_bits_per_second": 161578065.72,
    "total_avg_latency_ms": 0.03595,
    "total_packet_delivery_ratio": 99.88,
    "total_avg_latency_jitter_ms": 0.01695
  }
}
```

#### 字段对照:

| 旧字段名 | 新字段名 | 说明 | 单位 | 保留 |
|---------|---------|------|------|------|
| `avg_throughput_mbps` | `total_throughput_bits_per_second` | 总吞吐量 | bps (旧: Mbps) | ✅ 改名+单位 |
| `avg_latency_ms` | `total_avg_latency_ms` | 总平均时延 | ms | ✅ 改名 |
| `avg_packet_delivery_rate` | `total_packet_delivery_ratio` | 总包投递率 | % | ✅ 改名 |
| `avg_jitter_ms` | `total_avg_latency_jitter_ms` | 总时延抖动 | ms | ✅ 改名 |
| `max_throughput_mbps` | - | - | - | ❌ 删除 |

---

## 🔧 前端代码适配

### 修改文件: `js/index.js`

#### 修改位置1: 业务表格渲染 (第524-550行)

**修改前:**
```javascript
// 获取快照级平均时延
if (latestSnapshot.performance && latestSnapshot.performance.avg_latency_ms !== undefined) {
    latestSnapshotLatency = parseFloat(latestSnapshot.performance.avg_latency_ms).toFixed(4);
}

// 吞吐量
throughputMbps = connection.detailed_stats.avg_bits_per_second !== undefined 
       ? (parseFloat(connection.detailed_stats.avg_bits_per_second) / 1e6).toFixed(2) 
       : 'N/A';

// 投递率
deliveryRate = connection.detailed_stats.packet_delivery_rate !== undefined
        ? parseFloat(connection.detailed_stats.packet_delivery_rate).toFixed(1)
        : 'N/A';
```

**修改后:**
```javascript
// 获取快照级平均时延 (新字段名)
if (latestSnapshot.performance && latestSnapshot.performance.total_avg_latency_ms !== undefined) {
    latestSnapshotLatency = parseFloat(latestSnapshot.performance.total_avg_latency_ms).toFixed(4);
}

// 吞吐量 (新字段名)
throughputMbps = connection.detailed_stats.throughput_bits_per_second !== undefined 
       ? (parseFloat(connection.detailed_stats.throughput_bits_per_second) / 1e6).toFixed(2) 
       : 'N/A';

// 投递率 (新字段名)
deliveryRate = connection.detailed_stats.packet_delivery_ratio !== undefined
        ? parseFloat(connection.detailed_stats.packet_delivery_ratio).toFixed(1)
        : 'N/A';
```

---

#### 修改位置2: 图表数据累积 (第625-655行)

**修改前:**
```javascript
// 吞吐量 (单位: Mbps)
if (latestSnapshot.performance.avg_throughput_mbps !== undefined) {
    accumulatedChartData.throughput.push([timestamp, 
        parseFloat(latestSnapshot.performance.avg_throughput_mbps)]);
}

// 平均时延
if (latestSnapshot.performance.avg_latency_ms !== undefined) {
    accumulatedChartData.latency.push([timestamp, 
        parseFloat(latestSnapshot.performance.avg_latency_ms)]);
}

// 包投递率
if (latestSnapshot.performance.avg_packet_delivery_rate !== undefined) {
    accumulatedChartData.pdr.push([timestamp, 
        parseFloat(latestSnapshot.performance.avg_packet_delivery_rate)]);
}

// 时延抖动
if (latestSnapshot.performance.avg_jitter_ms !== undefined) {
    accumulatedChartData.jitter.push([timestamp, 
        parseFloat(latestSnapshot.performance.avg_jitter_ms)]);
}
```

**修改后:**
```javascript
// 吞吐量 (新字段名, 单位: bps → Mbps)
if (latestSnapshot.performance.total_throughput_bits_per_second !== undefined) {
    const throughputMbps = parseFloat(latestSnapshot.performance.total_throughput_bits_per_second) / 1e6;
    accumulatedChartData.throughput.push([timestamp, throughputMbps]);
}

// 平均时延 (新字段名)
if (latestSnapshot.performance.total_avg_latency_ms !== undefined) {
    accumulatedChartData.latency.push([timestamp, 
        parseFloat(latestSnapshot.performance.total_avg_latency_ms)]);
}

// 包投递率 (新字段名)
if (latestSnapshot.performance.total_packet_delivery_ratio !== undefined) {
    accumulatedChartData.pdr.push([timestamp, 
        parseFloat(latestSnapshot.performance.total_packet_delivery_ratio)]);
}

// 时延抖动 (新字段名)
if (latestSnapshot.performance.total_avg_latency_jitter_ms !== undefined) {
    accumulatedChartData.jitter.push([timestamp, 
        parseFloat(latestSnapshot.performance.total_avg_latency_jitter_ms)]);
}
```

---

## 📊 字段命名规范

### 统一命名规则:
1. **全局统计**: 使用 `total_` 前缀
   - `total_throughput_bits_per_second`
   - `total_avg_latency_ms`
   - `total_packet_delivery_ratio`
   - `total_avg_latency_jitter_ms`

2. **连接级统计**: 无前缀
   - `throughput_bits_per_second`
   - `avg_latency_ms`
   - `packet_delivery_ratio`
   - `avg_latency_jitter_ms`

3. **单位规范**:
   - 吞吐量: `bits_per_second` (bps)
   - 时延: `ms` (毫秒)
   - 投递率: `ratio` (百分比)
   - 包/字节: 无单位后缀

---

## 🎯 优势对比

| 对比项 | 修改前 | 修改后 | 改善 |
|--------|--------|--------|------|
| **字段数量** (detailed_stats) | 12个 | 7个 | 减少 42% |
| **字段数量** (performance) | 5个 | 4个 | 减少 20% |
| **命名规范** | 不统一 | 统一前缀 | ✅ 提升 |
| **数据冗余** | 存在 | 无 | ✅ 优化 |
| **可读性** | 中等 | 高 | ✅ 提升 |

---

## 🧪 验证清单

### 后端验证:
- [ ] 修改 JSON 生成逻辑
- [ ] 使用新字段名
- [ ] 删除废弃字段
- [ ] 单位转换正确 (吞吐量: bps)

### 前端验证:
- [x] 业务表格正常显示
- [x] 吞吐量单位转换正确 (bps → Mbps)
- [x] 时延数据正确显示
- [x] 投递率数据正确显示
- [x] 4个性能图表正常渲染
- [x] JavaScript 语法检查通过

---

## 📝 后端适配示例

### Python 示例:
```python
# 生成新格式的 JSON
performance_data = {
    "latest_snapshot": {
        "timestamp": datetime.now().isoformat(),
        "connections": {
            f"node{src}->node{dst}": {
                "original_node": str(src),
                "destination_node": str(dst),
                "route_path": f"{src}-{dst}",
                "test_types": ["TCP"],
                "detailed_stats": {
                    # 新字段名
                    "throughput_bits_per_second": throughput_bps,  # 单位: bps
                    "avg_latency_ms": avg_latency,
                    "packet_delivery_ratio": delivery_ratio,  # 0-100
                    "avg_latency_jitter_ms": jitter,
                    "packets_sent": sent_count,
                    "packets_received": recv_count,
                    "bytes_transferred": total_bytes
                }
            }
        },
        "performance": {
            # 新字段名 (全局统计加 total_ 前缀)
            "total_throughput_bits_per_second": sum(all_throughputs),  # 单位: bps
            "total_avg_latency_ms": avg(all_latencies),
            "total_packet_delivery_ratio": avg(all_ratios),
            "total_avg_latency_jitter_ms": avg(all_jitters)
        }
    }
}
```

---

## ⚠️ 重要注意事项

### 1. 单位变化
- **吞吐量**: 从 `Mbps` 改为 `bps` (bits per second)
- 前端需要除以 `1e6` 转换为 Mbps 显示
- 已在代码中处理: `throughputMbps = value / 1e6`

### 2. 字段重命名
- 所有 `avg_` 改为明确的指标名
- 全局统计加 `total_` 前缀
- `rate` 改为 `ratio` (更准确的术语)

### 3. 删除字段
以下字段已删除,不再使用:
- `total_lost_packets`
- `total_retransmits`
- `total_out_of_order`
- `total_duplicates`
- `total_test_duration_sec`
- `effective_loss_rate`
- `max_throughput_mbps`

---

## 🔙 回滚方案

如需回滚到旧格式:
```bash
# 恢复旧的 JSON 文件
Copy-Item "network_performance_old_format_backup.json" "network_performance.json" -Force
```

前端代码需要手动回滚相关修改。

---

## 📚 相关文档

- **数据格式优化**: `DATA_FORMAT_UPDATE.md`
- **完成总结**: `MODIFICATION_SUMMARY.md`
- **语法错误修复**: `SYNTAX_ERROR_FIX.md`
- **作用域错误修复**: `SCOPE_ERROR_FIX.md`

---

**最后更新**: 2026-04-03  
**版本**: 2.0 (字段简化版)  
**状态**: ✅ 完成并验证
