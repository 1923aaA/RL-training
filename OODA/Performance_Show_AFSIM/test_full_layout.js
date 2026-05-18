// 完整模拟实际JSON数据的布局计算

const fs = require('fs');
const clusterData = JSON.parse(fs.readFileSync('Cluster_info.json', 'utf8'));

// 按组网分组
const nodesByCluster = {};
clusterData.data.forEach(node => {
    const clusterName = node['组网信息'];
    if (!nodesByCluster[clusterName]) {
        nodesByCluster[clusterName] = [];
    }
    nodesByCluster[clusterName].push(node);
});

console.log('=== 实际数据布局计算 ===\n');

const clusterNames = Object.keys(nodesByCluster).sort();
const canvasWidth = 1200;
const canvasHeight = 800;
const padding = 50;
const totalClusters = clusterNames.length;

let gridCols = totalClusters <= 2 ? totalClusters : 2;
let gridRows = totalClusters <= 2 ? 1 : 2;

const cellWidth = (canvasWidth - 2 * padding) / gridCols;
const cellHeight = (canvasHeight - 2 * padding) / gridRows;

console.log(`组网数量: ${totalClusters}`);
console.log(`网格布局: ${gridRows}行 x ${gridCols}列`);
console.log(`单元格大小: ${cellWidth.toFixed(0)} x ${cellHeight.toFixed(0)}\n`);

clusterNames.forEach((clusterName, clusterIndex) => {
    const nodesInCluster = nodesByCluster[clusterName];
    const nodeCount = nodesInCluster.length;
    
    const gridRow = Math.floor(clusterIndex / gridCols);
    const gridCol = clusterIndex % gridCols;
    const centerX = padding + gridCol * cellWidth + cellWidth / 2;
    const centerY = padding + gridRow * cellHeight + cellHeight / 2;
    const radius = Math.min(cellWidth, cellHeight) * 0.35;
    
    console.log(`=== ${clusterName} (${nodeCount}个节点) ===`);
    console.log(`网格位置: 第${gridRow}行,第${gridCol}列`);
    console.log(`中心坐标: (${centerX.toFixed(1)}, ${centerY.toFixed(1)})`);
    console.log(`半径: ${radius.toFixed(1)}`);
    
    const hasCenterNode = nodesInCluster.some(n => n['角色'] === '中心节点');
    console.log(`有中心节点: ${hasCenterNode ? '是' : '否'}\n`);
    
    const positions = [];
    
    nodesInCluster.forEach((node, nodeIndex) => {
        let x, y, angle = null;
        
        if (node['角色'] === '中心节点') {
            x = centerX;
            y = centerY;
        } else {
            const surroundingNodeCount = hasCenterNode ? nodeCount - 1 : nodeCount;
            
            let surroundingIndex;
            if (hasCenterNode) {
                const centerNodeIndex = nodesInCluster.findIndex(n => n['角色'] === '中心节点');
                surroundingIndex = nodeIndex > centerNodeIndex ? nodeIndex - 1 : nodeIndex;
            } else {
                surroundingIndex = nodeIndex;
            }
            
            angle = (surroundingIndex * 2 * Math.PI) / surroundingNodeCount;
            x = centerX + radius * Math.cos(angle);
            y = centerY + radius * Math.sin(angle);
        }
        
        positions.push({
            id: node['节点标识符'],
            role: node['角色'],
            nodeIndex,
            angle,
            angleDeg: angle !== null ? (angle * 180 / Math.PI).toFixed(1) : null,
            x: x.toFixed(2),
            y: y.toFixed(2)
        });
    });
    
    // 打印节点位置
    positions.forEach(p => {
        if (p.angle === null) {
            console.log(`  节点${p.id}: 中心节点, 坐标=(${p.x}, ${p.y})`);
        } else {
            console.log(`  节点${p.id}: 角度=${p.angleDeg}°, 坐标=(${p.x}, ${p.y})`);
        }
    });
    
    // 检查是否有节点重合
    console.log('\n检查节点重合:');
    let hasOverlap = false;
    for (let i = 0; i < positions.length; i++) {
        for (let j = i + 1; j < positions.length; j++) {
            const d = Math.sqrt(
                Math.pow(parseFloat(positions[j].x) - parseFloat(positions[i].x), 2) +
                Math.pow(parseFloat(positions[j].y) - parseFloat(positions[i].y), 2)
            );
            if (d < 10) {  // 距离小于10px认为重合
                console.log(`  ❌ 节点${positions[i].id} 和 节点${positions[j].id} 重合! (距离: ${d.toFixed(2)}px)`);
                hasOverlap = true;
            }
        }
    }
    if (!hasOverlap) {
        console.log(`  ✅ 无节点重合`);
    }
    
    // 检查首尾节点
    const surroundingNodes = positions.filter(p => p.angle !== null);
    if (surroundingNodes.length >= 2) {
        const first = surroundingNodes[0];
        const last = surroundingNodes[surroundingNodes.length - 1];
        const distance = Math.sqrt(
            Math.pow(parseFloat(last.x) - parseFloat(first.x), 2) +
            Math.pow(parseFloat(last.y) - parseFloat(first.y), 2)
        );
        console.log(`\n首尾围绕节点: 节点${first.id} (${first.angleDeg}°) <-> 节点${last.id} (${last.angleDeg}°)`);
        console.log(`距离: ${distance.toFixed(2)}px`);
        console.log(distance < 20 ? '❌ 距离太近!' : '✅ 距离合理');
    }
    
    console.log('\n');
});
