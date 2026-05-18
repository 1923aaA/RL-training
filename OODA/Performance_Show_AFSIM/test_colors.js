const CLUSTER_COLORS = {
    "组网 1": "#FF4949",
    "组网 2": "#FFA74D",
    "组网 3": "#77CD63",
    "组网 4": "#44AFF0"
};

console.log('CLUSTER_COLORS 的键:');
Object.keys(CLUSTER_COLORS).forEach(key => {
    console.log(`  "${key}" (长度: ${key.length})`);
    console.log(`  Unicode 码点:`, [...key].map(c => `${c}(U+${c.charCodeAt(0).toString(16).toUpperCase().padStart(4, '0')})`).join(' '));
});

const testKey = "组网 1";
console.log(`\n测试键 "${testKey}":`);
console.log(`  长度: ${testKey.length}`);
console.log(`  Unicode 码点:`, [...testKey].map(c => `${c}(U+${c.charCodeAt(0).toString(16).toUpperCase().padStart(4, '0')})`).join(' '));
console.log(`  匹配结果:`, CLUSTER_COLORS[testKey]);
