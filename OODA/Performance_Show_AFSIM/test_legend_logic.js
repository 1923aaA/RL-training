const fs = require('fs');

// 从 index.js 复制的常量
const CLUSTER_COLORS = {
    "组网 1": "#FF4949",
    "组网 2": "#FFA74D",
    "组网 3": "#77CD63",
    "组网 4": "#44AFF0"
};

const OFFLINE_COLOR = 'rgba(128, 128, 128, 0.7)';

// 加载 Cluster_info.json
const clusterData = JSON.parse(fs.readFileSync('Cluster_info.json', 'utf8'));

console.log('=== 模拟图例生成逻辑 ===\n');

let categories = [];

// 首先添加活动的组网的图例项
if (clusterData && clusterData.metadata && clusterData.metadata.ranges && clusterData.metadata.ranges['组网信息']) {
    const sortedClusterNames = clusterData.metadata.ranges['组网信息'].slice().sort();
    console.log('metadata.ranges[组网信息]:', sortedClusterNames);
    
    sortedClusterNames.forEach(clusterNameFromMetadata => {
        console.log(`\n检查: "${clusterNameFromMetadata}"`);
        console.log(`  长度: ${clusterNameFromMetadata.length}`);
        console.log(`  Unicode:`, [...clusterNameFromMetadata].map(c => `${c}(U+${c.charCodeAt(0).toString(16).toUpperCase().padStart(4, '0')})`).join(' '));
        console.log(`  在 CLUSTER_COLORS 中?`, CLUSTER_COLORS[clusterNameFromMetadata] !== undefined);
        
        if (CLUSTER_COLORS[clusterNameFromMetadata]) {
            if (!categories.find(cat => cat.name === clusterNameFromMetadata)) {
                categories.push({
                    name: clusterNameFromMetadata,
                    itemStyle: { color: CLUSTER_COLORS[clusterNameFromMetadata] }
                });
                console.log(`  ✓ 添加到 categories`);
            }
        } else {
            console.log(`  ✗ 未添加 (不在 CLUSTER_COLORS 中)`);
        }
    });
}

// 检查数据中是否有退网节点
const hasOfflineNodes = clusterData.data.some(n => n['状态'] === '退网');
console.log(`\n是否有退网节点: ${hasOfflineNodes}`);

if (hasOfflineNodes) {
    if (!categories.find(cat => cat.name === '退网')) {
        categories.push({
            name: '退网',
            itemStyle: { color: OFFLINE_COLOR }
        });
        console.log('✓ 添加 "退网" 到 categories');
    }
}

console.log('\n=== 最终 categories ===');
categories.forEach((cat, i) => {
    console.log(`${i + 1}. "${cat.name}" - 颜色: ${cat.itemStyle.color}`);
});

console.log(`\n总计: ${categories.length} 个图例项`);
