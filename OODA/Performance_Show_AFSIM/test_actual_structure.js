// 模拟实际数据结构测试

function simulateActualData() {
    console.log('=== 模拟实际数据结构测试 ===\n');
    
    // 模拟组网1的节点 (8个节点: 1中心 + 1备用中心 + 6成员)
    const nodesInCluster = [
        { 节点标识符: 1, 角色: '中心节点', 组网信息: '组网1-TDMA' },
        { 节点标识符: 2, 角色: '备用中心节点', 组网信息: '组网1-TDMA' },
        { 节点标识符: 3, 角色: '成员节点', 组网信息: '组网1-TDMA' },
        { 节点标识符: 4, 角色: '成员节点', 组网信息: '组网1-TDMA' },
        { 节点标识符: 5, 角色: '成员节点', 组网信息: '组网1-TDMA' },
        { 节点标识符: 6, 角色: '成员节点', 组网信息: '组网1-TDMA' },
        { 节点标识符: 7, 角色: '成员节点', 组网信息: '组网1-TDMA' },
        { 节点标识符: 8, 角色: '成员节点', 组网信息: '组网1-TDMA' }
    ];
    
    const centerX = 400;
    const centerY = 300;
    const radius = 150;
    const nodeCount = nodesInCluster.length;
    
    console.log(`节点总数: ${nodeCount}`);
    console.log(`中心节点位置: 索引 0\n`);
    
    // === 旧逻辑 (有bug) ===
    console.log('=== 旧逻辑 ===');
    const oldPositions = [];
    
    nodesInCluster.forEach((node, nodeIndex) => {
        let x, y;
        if (node['角色'] === '中心节点') {
            x = centerX;
            y = centerY;
            oldPositions.push({ id: node['节点标识符'], role: node['角色'], nodeIndex, x, y, angle: null });
        } else {
            const angleStep = (2 * Math.PI) / Math.max(nodeCount - 1, 1);
            const adjustedIndex = nodeIndex - 1; // 简单减1
            const angle = adjustedIndex * angleStep;
            x = centerX + radius * Math.cos(angle);
            y = centerY + radius * Math.sin(angle);
            oldPositions.push({ id: node['节点标识符'], role: node['角色'], nodeIndex, adjustedIndex, angle, angleDeg: (angle * 180 / Math.PI).toFixed(1), x: x.toFixed(2), y: y.toFixed(2) });
        }
    });
    
    console.log('节点位置:');
    oldPositions.forEach(p => {
        if (p.angle === null) {
            console.log(`  节点${p.id} (${p.role}): 中心位置`);
        } else {
            console.log(`  节点${p.id} (${p.role}): nodeIndex=${p.nodeIndex}, adjustedIndex=${p.adjustedIndex}, 角度=${p.angleDeg}°, 坐标=(${p.x}, ${p.y})`);
        }
    });
    
    // 检查首尾围绕节点
    const oldSurrounding = oldPositions.filter(p => p.angle !== null);
    if (oldSurrounding.length >= 2) {
        const first = oldSurrounding[0];
        const last = oldSurrounding[oldSurrounding.length - 1];
        const distance = Math.sqrt(Math.pow(parseFloat(last.x) - parseFloat(first.x), 2) + 
                                     Math.pow(parseFloat(last.y) - parseFloat(first.y), 2));
        console.log(`\n首尾围绕节点: 节点${first.id} <-> 节点${last.id}`);
        console.log(`距离: ${distance.toFixed(2)} px`);
        console.log(distance > 10 ? '✅ 未重合' : '❌ 重合!\n');
    }
    
    
    // === 新逻辑 (修复后) ===
    console.log('\n=== 新逻辑 ===');
    const newPositions = [];
    
    nodesInCluster.forEach((node, nodeIndex) => {
        let x, y;
        if (node['角色'] === '中心节点') {
            x = centerX;
            y = centerY;
            newPositions.push({ id: node['节点标识符'], role: node['角色'], nodeIndex, x, y, angle: null });
        } else {
            const hasCenterNode = nodesInCluster.some(n => n['角色'] === '中心节点');
            const surroundingNodeCount = hasCenterNode ? nodeCount - 1 : nodeCount;
            
            const centerNodeIndex = nodesInCluster.findIndex(n => n['角色'] === '中心节点');
            const surroundingIndex = nodeIndex > centerNodeIndex ? nodeIndex - 1 : nodeIndex;
            
            const angle = (surroundingIndex * 2 * Math.PI) / surroundingNodeCount;
            x = centerX + radius * Math.cos(angle);
            y = centerY + radius * Math.sin(angle);
            newPositions.push({ id: node['节点标识符'], role: node['角色'], nodeIndex, surroundingIndex, angle, angleDeg: (angle * 180 / Math.PI).toFixed(1), x: x.toFixed(2), y: y.toFixed(2) });
        }
    });
    
    console.log('节点位置:');
    newPositions.forEach(p => {
        if (p.angle === null) {
            console.log(`  节点${p.id} (${p.role}): 中心位置`);
        } else {
            console.log(`  节点${p.id} (${p.role}): nodeIndex=${p.nodeIndex}, surroundingIndex=${p.surroundingIndex}, 角度=${p.angleDeg}°, 坐标=(${p.x}, ${p.y})`);
        }
    });
    
    // 检查首尾围绕节点
    const newSurrounding = newPositions.filter(p => p.angle !== null);
    if (newSurrounding.length >= 2) {
        const first = newSurrounding[0];
        const last = newSurrounding[newSurrounding.length - 1];
        const distance = Math.sqrt(Math.pow(parseFloat(last.x) - parseFloat(first.x), 2) + 
                                     Math.pow(parseFloat(last.y) - parseFloat(first.y), 2));
        console.log(`\n首尾围绕节点: 节点${first.id} <-> 节点${last.id}`);
        console.log(`距离: ${distance.toFixed(2)} px`);
        console.log(distance > 10 ? '✅ 未重合' : '❌ 重合!\n');
    }
}

simulateActualData();
