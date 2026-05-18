// 对比旧逻辑和新逻辑 - 无中心节点的情况

console.log('=== 对比: 10个节点,全部成员节点(无中心节点) ===\n');

const centerX = 400;
const centerY = 300;
const radius = 150;
const nodeCount = 10;

// 旧逻辑 (BUG版本)
console.log('旧逻辑 (有bug):');
const angleStep = (2 * Math.PI) / Math.max(nodeCount - 1, 1);  // BUG: 除以(nodeCount-1)
const oldPositions = [];

for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
    const adjustedIndex = nodeIndex;  // 无中心节点时 adjustedIndex = nodeIndex
    const angle = adjustedIndex * angleStep;
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);
    oldPositions.push({
        index: nodeIndex,
        angle,
        angleDeg: (angle * 180 / Math.PI).toFixed(1),
        x: x.toFixed(2),
        y: y.toFixed(2)
    });
}

oldPositions.forEach(p => {
    console.log(`  节点${p.index}: 角度=${p.angleDeg}°, 坐标=(${p.x}, ${p.y})`);
});

// 检查首尾节点
const oldFirst = oldPositions[0];
const oldLast = oldPositions[oldPositions.length - 1];
const oldDistance = Math.sqrt(
    Math.pow(parseFloat(oldLast.x) - parseFloat(oldFirst.x), 2) +
    Math.pow(parseFloat(oldLast.y) - parseFloat(oldFirst.y), 2)
);

console.log(`\n首尾节点: 节点0 (${oldFirst.angleDeg}°) <-> 节点${nodeCount-1} (${oldLast.angleDeg}°)`);
console.log(`距离: ${oldDistance.toFixed(2)}px`);
console.log(oldDistance < 20 ? '❌ 距离太近,视觉重合!' : '✅ 距离合理');

// 新逻辑 (修复后)
console.log('\n新逻辑 (已修复):');
const surroundingNodeCount = nodeCount;  // 无中心节点时 = nodeCount
const newPositions = [];

for (let nodeIndex = 0; nodeIndex < nodeCount; nodeIndex++) {
    const surroundingIndex = nodeIndex;
    const angle = (surroundingIndex * 2 * Math.PI) / surroundingNodeCount;  // 修复: 除以nodeCount
    const x = centerX + radius * Math.cos(angle);
    const y = centerY + radius * Math.sin(angle);
    newPositions.push({
        index: nodeIndex,
        angle,
        angleDeg: (angle * 180 / Math.PI).toFixed(1),
        x: x.toFixed(2),
        y: y.toFixed(2)
    });
}

newPositions.forEach(p => {
    console.log(`  节点${p.index}: 角度=${p.angleDeg}°, 坐标=(${p.x}, ${p.y})`);
});

const newFirst = newPositions[0];
const newLast = newPositions[newPositions.length - 1];
const newDistance = Math.sqrt(
    Math.pow(parseFloat(newLast.x) - parseFloat(newFirst.x), 2) +
    Math.pow(parseFloat(newLast.y) - parseFloat(newFirst.y), 2)
);

console.log(`\n首尾节点: 节点0 (${newFirst.angleDeg}°) <-> 节点${nodeCount-1} (${newLast.angleDeg}°)`);
console.log(`距离: ${newDistance.toFixed(2)}px`);
console.log(newDistance < 20 ? '❌ 距离太近,视觉重合!' : '✅ 距离合理');

console.log(`\n改进幅度: ${((newDistance - oldDistance) / oldDistance * 100).toFixed(1)}%`);
