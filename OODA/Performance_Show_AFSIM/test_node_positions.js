// 测试节点位置计算 - 验证首尾节点不重合

function testNodePositions() {
    console.log('=== 节点位置计算测试 ===\n');
    
    const centerX = 400;
    const centerY = 300;
    const radius = 150;
    
    // 测试场景1: 有中心节点,共8个节点 (1个中心 + 7个围绕)
    console.log('场景1: 8个节点 (1中心 + 7围绕)');
    const nodeCount1 = 8;
    const surroundingCount1 = 7;
    
    const positions1 = [];
    for (let i = 0; i < surroundingCount1; i++) {
        const angle = (i * 2 * Math.PI) / surroundingCount1;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        positions1.push({ index: i, angle: angle, angleDeg: (angle * 180 / Math.PI).toFixed(2), x: x.toFixed(2), y: y.toFixed(2) });
    }
    
    console.log('围绕节点位置:');
    positions1.forEach(p => {
        console.log(`  节点${p.index}: 角度=${p.angleDeg}°, 坐标=(${p.x}, ${p.y})`);
    });
    
    // 检查首尾节点距离
    const first1 = positions1[0];
    const last1 = positions1[positions1.length - 1];
    const distance1 = Math.sqrt(Math.pow(parseFloat(last1.x) - parseFloat(first1.x), 2) + 
                                 Math.pow(parseFloat(last1.y) - parseFloat(first1.y), 2));
    console.log(`\n首尾节点距离: ${distance1.toFixed(2)} px`);
    console.log(`理论最小距离: ${(2 * radius * Math.sin(Math.PI / surroundingCount1)).toFixed(2)} px`);
    console.log(distance1 > 10 ? '✅ 首尾节点未重合' : '❌ 首尾节点重合!\n');
    
    
    // 测试场景2: 旧逻辑 (有bug的版本)
    console.log('\n场景2: 旧逻辑 (bug版本) - 8个节点');
    const positions2 = [];
    const angleStep = (2 * Math.PI) / (nodeCount1 - 1); // 旧逻辑: 除以(nodeCount-1)
    
    for (let i = 0; i < surroundingCount1; i++) {
        const angle = i * angleStep;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        positions2.push({ index: i, angle: angle, angleDeg: (angle * 180 / Math.PI).toFixed(2), x: x.toFixed(2), y: y.toFixed(2) });
    }
    
    console.log('围绕节点位置:');
    positions2.forEach(p => {
        console.log(`  节点${p.index}: 角度=${p.angleDeg}°, 坐标=(${p.x}, ${p.y})`);
    });
    
    const first2 = positions2[0];
    const last2 = positions2[positions2.length - 1];
    const distance2 = Math.sqrt(Math.pow(parseFloat(last2.x) - parseFloat(first2.x), 2) + 
                                 Math.pow(parseFloat(last2.y) - parseFloat(first2.y), 2));
    console.log(`\n首尾节点距离: ${distance2.toFixed(2)} px`);
    console.log(distance2 > 10 ? '✅ 首尾节点未重合' : '❌ 首尾节点重合 (BUG)!\n');
}

testNodePositions();
