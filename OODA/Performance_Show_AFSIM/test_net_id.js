// 验证 cluster_id 到 net_id 的修改

const fs = require('fs');
const linkData = JSON.parse(fs.readFileSync('Link_info.json', 'utf8'));

console.log('=== Link_info.json 字段验证 ===\n');

console.log('clusters 数量:', linkData.clusters.length);

linkData.clusters.forEach((cluster, index) => {
    console.log(`\n组网 ${index + 1}:`);
    
    // 检查是否有 net_id
    if (cluster.net_id !== undefined) {
        console.log(`  ✅ net_id: ${cluster.net_id}`);
    } else {
        console.log(`  ❌ net_id 未定义`);
    }
    
    // 检查是否还有旧的 cluster_id
    if (cluster.cluster_id !== undefined) {
        console.log(`  ⚠️  cluster_id 仍存在: ${cluster.cluster_id} (应该删除)`);
    } else {
        console.log(`  ✅ cluster_id 已删除`);
    }
    
    console.log(`  head_id: ${cluster.head_id}`);
    console.log(`  backup_head_id: ${cluster.backup_head_id}`);
    console.log(`  nodes 数量: ${cluster.nodes.length}`);
    console.log(`  nodes: [${cluster.nodes.join(', ')}]`);
});

console.log('\n=== 验证结果 ===');
const hasOldField = linkData.clusters.some(c => c.cluster_id !== undefined);
const hasNewField = linkData.clusters.every(c => c.net_id !== undefined);

if (!hasOldField && hasNewField) {
    console.log('✅ 字段重命名成功: cluster_id → net_id');
} else {
    if (hasOldField) console.log('❌ 仍存在 cluster_id 字段');
    if (!hasNewField) console.log('❌ 缺少 net_id 字段');
}
