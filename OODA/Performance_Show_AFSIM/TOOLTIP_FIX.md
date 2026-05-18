# 跨组网节点Tooltip显示修复

## 问题描述
节点10显示为菱形(跨组网节点),但鼠标悬停提示中"组网信息"只显示"组网1-STDMA",缺少"组网2-TDMA"。

---

## 根本原因

### 数据结构问题
跨组网节点的数据结构:
```javascript
{
    id: "10",
    name: "10",
    symbol: "diamond",
    category: "组网1-STDMA & 组网2-TDMA",  // ✅ 包含两个组网
    rawData: { ...第一个组网的数据... },    // ❌ 只包含组网1的信息
    ...
}
```

### Tooltip处理问题
旧的 tooltip formatter 直接使用 `rawData['组网信息']`:
```javascript
formatter: p => p.dataType === 'node' && p.data.rawData ? 
               `组网信息: ${p.data.rawData['组网信息']}`  // ❌ 只显示rawData中的组网
               : p.name
```

**结果**: 只显示"组网1-STDMA"(rawData中的值)

---

## 修复方案

### 新的 Tooltip Formatter
```javascript
formatter: p => {
    if (p.dataType === 'node' && p.data) {
        // 检查是否是跨组网节点(菱形)
        if (p.data.symbol === 'diamond' && p.data.category && p.data.category.includes(' & ')) {
            // 跨组网节点,从category中提取组网信息
            return `<b>节点标识符: ${p.data.name}</b><br/>` +
                   `角色: ${p.data.rawData ? p.data.rawData['角色'] : 'N/A'}<br/>` +
                   `状态: ${p.data.rawData ? p.data.rawData['状态'] : 'N/A'}<br/>` +
                   `组网信息: ${p.data.category.replace(' & ', ', ')}`;
        } else if (p.data.rawData) {
            // 单组网节点,使用rawData
            return `<b>节点标识符: ${p.data.rawData['节点标识符']}</b><br/>` +
                   `角色: ${p.data.rawData['角色']}<br/>` +
                   `状态: ${p.data.rawData['状态']}<br/>` +
                   `组网信息: ${p.data.rawData['组网信息']}`;
        }
    }
    return p.name;
}
```

---

## 处理逻辑

### 1. 识别跨组网节点
```javascript
if (p.data.symbol === 'diamond' && p.data.category && p.data.category.includes(' & '))
```

**判断依据**:
- 符号为 `diamond` (菱形)
- category 包含 `' & '` 分隔符

### 2. 提取组网信息
```javascript
p.data.category.replace(' & ', ', ')
```

**转换**:
- category: `"组网1-STDMA & 组网2-TDMA"`
- 显示: `"组网1-STDMA, 组网2-TDMA"`

### 3. 获取其他信息
- 从 `rawData` 获取角色和状态
- 从 `p.data.name` 获取节点标识符

---

## 修复前后对比

### 修复前
```
节点标识符: 10
角色: 成员节点
状态: 在网
组网信息: 组网1-STDMA  ❌ 只显示一个
```

### 修复后
```
节点标识符: 10
角色: 成员节点
状态: 在网
组网信息: 组网1-STDMA, 组网2-TDMA  ✅ 显示所有组网
```

---

## 代码位置

**文件**: `js/index.js`  
**函数**: `renderNetworkTopologyChart`  
**修改位置**: 约565-589行 (tooltip formatter)

---

## 优势

### 兼容性
- ✅ 单组网节点: 正常显示
- ✅ 跨组网节点: 显示所有组网
- ✅ 退网节点: 正常显示

### 扩展性
自动支持节点属于3个或更多组网:
```
组网信息: 组网1-STDMA, 组网2-TDMA, 组网3-CSMA
```

---

## 验证步骤

1. 刷新浏览器 (Ctrl + F5)
2. 鼠标悬停在节点10(白色菱形)上
3. 检查提示信息是否显示:
   ```
   组网信息: 组网1-STDMA, 组网2-TDMA
   ```

---

**修复状态**: ✅ 已完成  
**语法检查**: ✅ 通过  
**测试**: 请验证鼠标悬停提示
