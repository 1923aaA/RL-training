// 完整模拟前端渲染逻辑,检查节点10是否会被渲染

const fs = require('fs');
const clusterData = JSON.parse(fs.readFileSync('Cluster_info.json', 'utf8'));

// 辅助函数
function extractClusterNumber(clusterInfo) {
    if (!clusterInfo) return null;
    const match = clusterInfo.match(/组网(\d+)/);
    return match ? match[1] : null;
}

const CLUSTER_COLORS = {
    "1": "#FF4949",
    "2": "#FFA74D",
    "3": "#77CD63",
    "4": "#44AFF0"
};

function getClusterColor(clusterInfo) {
    const clusterNumber = extractClusterNumber(clusterInfo);
    return CLUSTER_COLORS[clusterNumber] || '#ccc';
}

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

console.log('=== 完整渲染逻辑模拟 ===\n');

// 节点处理
const nodeClusterMap = {};
const offlineNodes = [];
let nodes = [];

clusterData.data.forEach(node => {
    const nodeId = node['节点标识符'];
    if (node['状态'] === '退网') {
        offlineNodes.push(node);
    } else {
        if (!nodeClusterMap[nodeId]) {
            nodeClusterMap[nodeId] = [];
        }
        nodeClusterMap[nodeId].push({
            clusterName: node['组网信息'],
            node: node
        });
    }
});

const crossClusterNodes = [];
const singleClusterNodes = {};

Object.keys(nodeClusterMap).forEach(nodeId => {
    const clusters = nodeClusterMap[nodeId];
    if (clusters.length > 1) {
        crossClusterNodes.push({
            nodeId: parseInt(nodeId),
            clusters: clusters,
            node: clusters[0].node
        });
    } else {
        const clusterName = clusters[0].clusterName;
        if (!singleClusterNodes[clusterName]) {
            singleClusterNodes[clusterName] = [];
        }
        singleClusterNodes[clusterName].push(clusters[0].node);
    }
});

console.log('节点分类结果:');
console.log(`  单组网节点数: ${Object.values(singleClusterNodes).flat().length}`);
console.log(`  跨组网节点数: ${crossClusterNodes.length}`);
console.log(`  跨组网节点ID: ${crossClusterNodes.map(n => n.nodeId).join(', ')}\n`);

// 布局计算
const clusterNames = Object.keys(singleClusterNodes).sort();
const canvasWidth = 1200;
const canvasHeight = 800;
const padding = 50;
const totalClusters = clusterNames.length;
const gridCols = totalClusters <= 2 ? totalClusters : 2;
const gridRows = totalClusters <= 2 ? 1 : 2;
const cellWidth = (canvasWidth - 2 * padding) / gridCols;
const cellHeight = (canvasHeight - 2 * padding) / gridRows;
const clusterCenters = {};

console.log('组网布局:');
clusterNames.forEach((clusterName, clusterIndex) => {
    const gridRow = Math.floor(clusterIndex / gridCols);
    const gridCol = clusterIndex % gridCols;
    const centerX = padding + gridCol * cellWidth + cellWidth / 2;
    const centerY = padding + gridRow * cellHeight + cellHeight / 2;
    clusterCenters[clusterName] = { centerX, centerY, gridRow, gridCol };
    console.log(`  ${clusterName}: 中心(${centerX.toFixed(1)}, ${centerY.toFixed(1)})`);
});

// 处理跨组网节点
console.log('\n处理跨组网节点:');
crossClusterNodes.forEach(crossNode => {
    const { nodeId, clusters, node } = crossNode;
    const clusterNameList = clusters.map(c => c.clusterName).sort();
    
    let x, y;
    if (clusterNameList.length === 2) {
        const center1 = clusterCenters[clusterNameList[0]];
        const center2 = clusterCenters[clusterNameList[1]];
        
        if (center1 && center2) {
            x = (center1.centerX + center2.centerX) / 2;
            y = (center1.centerY + center2.centerY) / 2;
        } else {
            x = canvasWidth / 2;
            y = canvasHeight / 2;
        }
    }
    
    const colors = clusterNameList.map(cname => getClusterColor(cname));
    const mixedColor = mixColors(colors);
    
    console.log(`  节点${nodeId}:`);
    console.log(`    所属组网: ${clusterNameList.join(', ')}`);
    console.log(`    位置: (${x.toFixed(1)}, ${y.toFixed(1)})`);
    console.log(`    颜色: ${mixedColor}`);
    console.log(`    符号: 菱形`);
    
    nodes.push({
        id: String(nodeId),
        name: String(nodeId),
        symbol: 'diamond',
        x: x,
        y: y,
        itemStyle: { 
            color: mixedColor,
            borderColor: '#fff',
            borderWidth: 2
        },
        category: clusterNameList.join(' & ')
    });
});

console.log(`\n总节点数(nodes数组): ${nodes.length}`);
console.log(`跨组网节点在nodes中: ${nodes.filter(n => n.symbol === 'diamond').length}个`);

// 检查节点10是否在nodes数组中
const node10 = nodes.find(n => n.id === '10');
if (node10) {
    console.log('\n✅ 节点10已添加到nodes数组:');
    console.log(`   ID: ${node10.id}`);
    console.log(`   位置: (${node10.x}, ${node10.y})`);
    console.log(`   颜色: ${node10.itemStyle.color}`);
    console.log(`   符号: ${node10.symbol}`);
    console.log(`   类别: ${node10.category}`);
} else {
    console.log('\n❌ 节点10未添加到nodes数组!');
}
