// 测试跨组网节点处理逻辑

const fs = require('fs');
const clusterData = JSON.parse(fs.readFileSync('Cluster_info.json', 'utf8'));

console.log('=== 跨组网节点测试 ===\n');

// 模拟前端逻辑: 识别跨组网节点
const nodeClusterMap = {};

clusterData.data.forEach(node => {
    const nodeId = node['节点标识符'];
    if (node['状态'] !== '退网') {
        if (!nodeClusterMap[nodeId]) {
            nodeClusterMap[nodeId] = [];
        }
        nodeClusterMap[nodeId].push({
            clusterName: node['组网信息'],
            node: node
        });
    }
});

// 分类节点
const crossClusterNodes = [];
const singleClusterNodes = {};

Object.keys(nodeClusterMap).forEach(nodeId => {
    const clusters = nodeClusterMap[nodeId];
    if (clusters.length > 1) {
        crossClusterNodes.push({
            nodeId: parseInt(nodeId),
            clusters: clusters.map(c => c.clusterName)
        });
    } else {
        const clusterName = clusters[0].clusterName;
        if (!singleClusterNodes[clusterName]) {
            singleClusterNodes[clusterName] = [];
        }
        singleClusterNodes[clusterName].push(parseInt(nodeId));
    }
});

console.log('单组网节点统计:');
Object.keys(singleClusterNodes).sort().forEach(clusterName => {
    console.log(`  ${clusterName}: ${singleClusterNodes[clusterName].length}个节点`);
    console.log(`    节点ID: ${singleClusterNodes[clusterName].join(', ')}`);
});

console.log(`\n跨组网节点统计: ${crossClusterNodes.length}个`);
crossClusterNodes.forEach(crossNode => {
    console.log(`  节点${crossNode.nodeId}: 属于 ${crossNode.clusters.join(' & ')}`);
});

// 测试颜色混合
function mixColors(colors) {
    if (colors.length === 0) return '#ccc';
    if (colors.length === 1) return colors[0];
    
    let totalR = 0, totalG = 0, totalB = 0;
    
    colors.forEach(color => {
        const hex = color.replace('#', '');
        const r = parseInt(hex.substring(0, 2), 16);
        const g = parseInt(hex.substring(2, 4), 16);
        const b = parseInt(hex.substring(4, 6), 16);
        totalR += r;
        totalG += g;
        totalB += b;
    });
    
    const avgR = Math.round(totalR / colors.length);
    const avgG = Math.round(totalG / colors.length);
    const avgB = Math.round(totalB / colors.length);
    
    return `#${avgR.toString(16).padStart(2, '0')}${avgG.toString(16).padStart(2, '0')}${avgB.toString(16).padStart(2, '0')}`;
}

console.log('\n颜色混合测试:');
const color1 = '#FF4949'; // 组网1 - 红色
const color2 = '#FFA74D'; // 组网2 - 橙色
const mixed = mixColors([color1, color2]);
console.log(`  组网1颜色: ${color1}`);
console.log(`  组网2颜色: ${color2}`);
console.log(`  混合颜色: ${mixed}`);

// 测试位置计算
console.log('\n位置计算测试:');
const canvasWidth = 1200;
const canvasHeight = 800;
const padding = 50;
const gridCols = 2;
const cellWidth = (canvasWidth - 2 * padding) / gridCols;
const cellHeight = (canvasHeight - 2 * padding) / 1;

// 组网1 (第0列)
const center1 = {
    centerX: padding + 0 * cellWidth + cellWidth / 2,
    centerY: padding + 0 * cellHeight + cellHeight / 2
};

// 组网2 (第1列)
const center2 = {
    centerX: padding + 1 * cellWidth + cellWidth / 2,
    centerY: padding + 0 * cellHeight + cellHeight / 2
};

// 跨组网节点位置 (两个中心的中点)
const crossX = (center1.centerX + center2.centerX) / 2;
const crossY = (center1.centerY + center2.centerY) / 2;

console.log(`  组网1中心: (${center1.centerX.toFixed(1)}, ${center1.centerY.toFixed(1)})`);
console.log(`  组网2中心: (${center2.centerX.toFixed(1)}, ${center2.centerY.toFixed(1)})`);
console.log(`  跨组网节点10位置: (${crossX.toFixed(1)}, ${crossY.toFixed(1)})`);
console.log(`  位置描述: 在两个组网区域的交界处(画布中央)`);
