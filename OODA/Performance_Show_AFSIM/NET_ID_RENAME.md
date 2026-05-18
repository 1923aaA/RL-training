# cluster_id → net_id 字段重命名

## 修改概述

将 `Link_info.json` 中的 `cluster_id` 字段统一重命名为 `net_id`,以与项目中其他地方使用的"组网"概念保持一致。

## 修改内容

### 1. Link_info.json
**变更**: 将所有 `cluster_id` 字段重命名为 `net_id`

**修改前**:
```json
{
  "clusters": [
    {
      "cluster_id": 1,
      "head_id": 1,
      ...
    },
    {
      "cluster_id": 2,
      "head_id": 11,
      ...
    }
  ]
}
```

**修改后**:
```json
{
  "clusters": [
    {
      "net_id": 1,
      "head_id": 1,
      ...
    },
    {
      "net_id": 2,
      "head_id": 11,
      ...
    }
  ]
}
```

### 2. js/index.js (Line 533)
**变更**: 更新警告日志中对该字段的引用

**修改前**:
```javascript
console.warn(`组网内连接警告: 节点ID未定义，组网 ${cl.cluster_id}, 索引 ${i} 或 ${j}`);
```

**修改后**:
```javascript
console.warn(`组网内连接警告: 节点ID未定义，组网 ${cl.net_id}, 索引 ${i} 或 ${j}`);
```

## 影响范围

1. **数据文件**: `Link_info.json` 的数据结构
2. **前端代码**: `js/index.js` 中的一处日志输出

## 验证步骤

1. **JSON 格式验证**:
   ```bash
   node -e "JSON.parse(require('fs').readFileSync('Link_info.json', 'utf8'))"
   ```
   结果: ✅ 通过

2. **JavaScript 语法验证**:
   ```bash
   node -c js/index.js
   ```
   结果: ✅ 通过

3. **字段重命名验证**:
   ```bash
   node test_net_id.js
   ```
   结果: 
   - ✅ net_id 字段存在于所有组网
   - ✅ cluster_id 字段已完全移除

## 修改理由

1. **术语统一**: 项目中已将"簇"统一改为"组网",对应的字段名也应保持一致
2. **代码可读性**: `net_id` 比 `cluster_id` 更直观地表达"组网ID"的含义
3. **维护性**: 统一的命名约定便于后续开发和维护

## 注意事项

如果后端代码或其他系统也使用 `Link_info.json`,需要同步更新这些系统中对该字段的引用。

## 测试文件

创建了 `test_net_id.js` 用于验证字段重命名的正确性。
