# 组网信息动态协议支持说明

## 修改时间
2026-04-03

## 修改目的
支持动态的组网协议类型,使组网信息从固定的"组网 1"格式改为"组网N-协议名"格式(如"组网1-TDMA"),其中协议名可由外部动态填入。

---

## 数据格式变化

### 旧格式
```json
"组网信息": ["组网 1", "组网 2", "组网 3", "组网 4"]
```

### 新格式
```json
"组网信息": ["组网1-TDMA", "组网2-SPMA", "组网3-CSMA", "组网4-FFF"]
```

**格式规则:**
- **固定部分**: `组网N-` (其中 N 为数字 1-4)
- **动态部分**: 协议名称 (如 TDMA, SPMA, CSMA, FFF 等)
- **无空格**: 移除了"组网"和数字之间的空格

---

## 前端代码修改

### 1. 颜色映射重构 (js/index.js 约 36-62 行)

#### 旧方式
```javascript
const CLUSTER_COLORS = {
    "组网 1": "#FF4949",
    "组网 2": "#FFA74D",
    "组网 3": "#77CD63",
    "组网 4": "#44AFF0"
};
```

#### 新方式
```javascript
// 使用组网编号作为键
const CLUSTER_COLORS = {
    "1": "#FF4949", // 红色 - 组网1
    "2": "#FFA74D", // 橙色 - 组网2
    "3": "#77CD63", // 绿色 - 组网3
    "4": "#44AFF0"  // 蓝色 - 组网4
};

// 辅助函数: 从组网信息中提取组网编号
function extractClusterNumber(clusterInfo) {
    if (!clusterInfo) return null;
    const match = clusterInfo.match(/组网(\d+)/);
    return match ? match[1] : null;
}

// 辅助函数: 根据组网信息获取颜色
function getClusterColor(clusterInfo) {
    const clusterNumber = extractClusterNumber(clusterInfo);
    return CLUSTER_COLORS[clusterNumber] || '#ccc';
}
```

**优点:**
- 颜色映射与具体协议名解耦
- 支持任意协议名称,只需保持"组网N-"格式
- 自动回退到默认灰色(#ccc)处理未定义的组网编号

---

### 2. 节点渲染逻辑更新 (约 231 行)

#### 旧代码
```javascript
const baseColor = CLUSTER_COLORS[node['组网信息']] || '#ccc';
```

#### 新代码
```javascript
const baseColor = getClusterColor(node['组网信息']); // 使用辅助函数
```

---

### 3. 图例生成逻辑更新 (约 295-308 行)

#### 旧代码
```javascript
if (CLUSTER_COLORS[clusterNameFromMetadata]) {
    categories.push({
        name: clusterNameFromMetadata,
        itemStyle: { color: CLUSTER_COLORS[clusterNameFromMetadata] }
    });
}
```

#### 新代码
```javascript
const clusterNumber = extractClusterNumber(clusterNameFromMetadata);
if (clusterNumber && CLUSTER_COLORS[clusterNumber]) {
    categories.push({
        name: clusterNameFromMetadata, // 显示完整名称 "组网1-TDMA"
        itemStyle: { color: CLUSTER_COLORS[clusterNumber] } // 使用编号匹配颜色
    });
}
```

---

## 提取逻辑示例

### 正则表达式
```javascript
/组网(\d+)/
```

### 提取示例
```javascript
"组网1-TDMA"   → 提取编号: "1" → 颜色: #FF4949
"组网2-SPMA"   → 提取编号: "2" → 颜色: #FFA74D
"组网3-CSMA"   → 提取编号: "3" → 颜色: #77CD63
"组网4-FFF"    → 提取编号: "4" → 颜色: #44AFF0
"组网5-XYZ"    → 提取编号: "5" → 颜色: #ccc (默认)
"invalid"      → 提取编号: null → 颜色: #ccc (默认)
```

---

## 动态更新协议名

### 后端数据生成示例

```python
# Python 示例
protocols = ["TDMA", "SPMA", "CSMA", "FFF"]
nodes = []

for i, protocol in enumerate(protocols, start=1):
    cluster_info = f"组网{i}-{protocol}"
    # 为该组网创建节点
    nodes.append({
        "节点标识符": node_id,
        "角色": role,
        "组网信息": cluster_info,  # 动态生成
        "状态": "在网"
    })
```

```javascript
// JavaScript 示例
const protocols = ["TDMA", "SPMA", "CSMA", "FFF"];
const nodes = [];

protocols.forEach((protocol, index) => {
    const clusterNumber = index + 1;
    const clusterInfo = `组网${clusterNumber}-${protocol}`;
    // 为该组网创建节点
    nodes.push({
        "节点标识符": nodeId,
        "角色": role,
        "组网信息": clusterInfo,  // 动态生成
        "状态": "在网"
    });
});
```

---

## 支持的协议名格式

### ✅ 有效格式
- `组网1-TDMA`
- `组网2-SPMA-V2`
- `组网3-CSMA/CD`
- `组网4-自定义协议`
- `组网1-123ABC` (包含数字和字母)

### ✅ 关键要求
1. 必须以 `组网` 开头
2. 紧跟数字编号 (1-9 或多位数)
3. 后面可以是任意字符作为协议名

### ❌ 无效格式
- `网络1-TDMA` (不是"组网"开头)
- `组网A-TDMA` (编号不是数字)
- `TDMA-组网1` (格式颠倒)

---

## 图例显示

### 显示效果
图例将显示完整的组网信息:
- 组网1-TDMA (红色圆点)
- 组网2-SPMA (橙色圆点)
- 组网3-CSMA (绿色圆点)
- 组网4-FFF (蓝色圆点)
- 退网 (灰色圆点)

### 鼠标悬停提示
```
节点标识符: 1
角色: 中心节点
状态: 在网
组网信息: 组网1-TDMA
```

---

## 扩展性

### 添加更多组网
如需支持更多组网(如组网5, 组网6等),只需:

1. **更新颜色映射**:
```javascript
const CLUSTER_COLORS = {
    "1": "#FF4949",
    "2": "#FFA74D",
    "3": "#77CD63",
    "4": "#44AFF0",
    "5": "#9C27B0", // 新增 - 紫色
    "6": "#00BCD4"  // 新增 - 青色
};
```

2. **更新 JSON 数据**:
```json
"组网信息": [
    "组网1-TDMA",
    "组网2-SPMA",
    "组网3-CSMA",
    "组网4-FFF",
    "组网5-新协议",
    "组网6-另一协议"
]
```

其他代码无需修改,自动支持!

---

## 测试验证

### 测试脚本
运行 `test_cluster_extraction.js` 验证提取逻辑:
```bash
node test_cluster_extraction.js
```

### 预期输出
```
=== 组网信息提取测试 ===

输入: "组网1-TDMA"
  提取编号: 1
  对应颜色: #FF4949
...
```

---

## 相关文件
- `Cluster_info.json` - 数据文件 (已更新)
- `js/index.js` - 前端代码 (已更新)
- `test_cluster_extraction.js` - 测试脚本 (新增)

---

## 优势总结

1. **灵活性**: 协议名可随时更改,无需修改前端代码
2. **扩展性**: 添加新组网只需更新颜色映射
3. **兼容性**: 自动处理未定义的组网编号
4. **可维护性**: 逻辑集中在辅助函数中
5. **清晰性**: 图例和提示显示完整的组网信息
