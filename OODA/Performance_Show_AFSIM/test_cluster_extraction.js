// 测试组网信息提取逻辑

// 复制辅助函数
function extractClusterNumber(clusterInfo) {
    if (!clusterInfo) return null;
    const match = clusterInfo.match(/组网(\d+)/);
    return match ? match[1] : null;
}

function getClusterColor(clusterInfo) {
    const CLUSTER_COLORS = {
        "1": "#FF4949",
        "2": "#FFA74D",
        "3": "#77CD63",
        "4": "#44AFF0"
    };
    const clusterNumber = extractClusterNumber(clusterInfo);
    return CLUSTER_COLORS[clusterNumber] || '#ccc';
}

// 测试用例
const testCases = [
    "组网1-TDMA",
    "组网2-SPMA",
    "组网3-CSMA",
    "组网4-FFF",
    "组网5-UNKNOWN",
    "invalid",
    null
];

console.log('=== 组网信息提取测试 ===\n');

testCases.forEach(testCase => {
    const number = extractClusterNumber(testCase);
    const color = getClusterColor(testCase);
    console.log(`输入: "${testCase}"`);
    console.log(`  提取编号: ${number}`);
    console.log(`  对应颜色: ${color}`);
    console.log('');
});

// 测试实际数据
const fs = require('fs');
const clusterData = JSON.parse(fs.readFileSync('Cluster_info.json', 'utf8'));

console.log('=== 实际数据测试 ===\n');
console.log('metadata.ranges.组网信息:');
clusterData.metadata.ranges['组网信息'].forEach(info => {
    const number = extractClusterNumber(info);
    const color = getClusterColor(info);
    console.log(`  "${info}" -> 编号: ${number}, 颜色: ${color}`);
});

console.log('\n实际数据中的组网信息 (去重):');
const uniqueClusterInfos = [...new Set(clusterData.data.map(n => n['组网信息']))].sort();
uniqueClusterInfos.forEach(info => {
    const number = extractClusterNumber(info);
    const color = getClusterColor(info);
    const count = clusterData.data.filter(n => n['组网信息'] === info).length;
    console.log(`  "${info}" -> 编号: ${number}, 颜色: ${color}, 节点数: ${count}`);
});
