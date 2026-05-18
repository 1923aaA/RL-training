// 测试没有中心节点的情况

function testNoCenterNode() {
    console.log('=== 测试: 10个节点,全部是成员节点(无中心节点) ===\n');
    
    const centerX = 400;
    const centerY = 300;
    const radius = 150;
    const nodeCount = 10;
    
    console.log('当前逻辑:');
    const positions = [];
    for (let i = 0; i < nodeCount; i++) {
        const angle = (i * 2 * Math.PI) / nodeCount;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        positions.push({ 
            index: i, 
            angle: angle, 
            angleDeg: (angle * 180 / Math.PI).toFixed(1), 
            x: x.toFixed(2), 
            y: y.toFixed(2) 
        });
    }
    
    positions.forEach(p => {
        console.log(`  节点${p.index}: 角度=${p.angleDeg}°, 坐标=(${p.x}, ${p.y})`);
    });
    
    // 检查首尾节点距离
    const first = positions[0];
    const last = positions[positions.length - 1];
    const distance = Math.sqrt(
        Math.pow(parseFloat(last.x) - parseFloat(first.x), 2) + 
        Math.pow(parseFloat(last.y) - parseFloat(first.y), 2)
    );
    
    console.log(`\n首尾节点:`);
    console.log(`  节点0: 角度=${first.angleDeg}°, 坐标=(${first.x}, ${first.y})`);
    console.log(`  节点${nodeCount-1}: 角度=${last.angleDeg}°, 坐标=(${last.x}, ${last.y})`);
    console.log(`  距离: ${distance.toFixed(2)} px`);
    
    // 理论相邻节点间距
    const neighborDistance = 2 * radius * Math.sin(Math.PI / nodeCount);
    console.log(`  理论相邻间距: ${neighborDistance.toFixed(2)} px`);
    console.log(`  实际/理论比: ${(distance / neighborDistance).toFixed(2)}`);
    
    if (distance < 20) {
        console.log(`❌ BUG: 首尾节点距离太近 (< 20px), 视觉上会重合!`);
    } else {
        console.log(`✅ 首尾节点距离合理`);
    }
    
    // 检查所有相邻节点间距
    console.log(`\n所有相邻节点间距:`);
    for (let i = 0; i < nodeCount; i++) {
        const next = (i + 1) % nodeCount;
        const d = Math.sqrt(
            Math.pow(parseFloat(positions[next].x) - parseFloat(positions[i].x), 2) + 
            Math.pow(parseFloat(positions[next].y) - parseFloat(positions[i].y), 2)
        );
        const warning = Math.abs(d - neighborDistance) > 1 ? ' ⚠️ 异常' : '';
        console.log(`  节点${i} -> 节点${next}: ${d.toFixed(2)} px${warning}`);
    }
}

testNoCenterNode();
