// ====================================================================================
// AFSIM 版 - 平台信息 + 协同效能统计
// ====================================================================================

// ====================================================================================
// 模块: 平台信息表格 (ID: produce)
// ====================================================================================
function loadPlatformInfo(platformDataToLoad) {
    const container = document.getElementById('produce');
    if (!container) return;

    if (!platformDataToLoad || !platformDataToLoad.platforms) {
        container.innerHTML = '<p style="color:#4E72B0;text-align:center;padding-top:20px;">暂无平台数据</p>';
        return;
    }

    try {
        container.innerHTML = '';
        const platforms = platformDataToLoad.platforms;

        const table = document.createElement('table');
        table.className = 'centertable topology-table';
        table.style.width = '100%';
        table.style.marginTop = '10px';

        const thead = table.createTHead();
        const headerRow = thead.insertRow();
        ['平台标识', '平台名称', '传感器', '平台类型', '阵营'].forEach(headerText => {
            const th = document.createElement('th');
            th.textContent = headerText;
            headerRow.appendChild(th);
        });

        const tbody = table.createTBody();
        if (platforms.length > 0) {
            platforms.forEach(p => {
                const row = tbody.insertRow();
                row.insertCell().textContent = p.platform_index !== undefined ? p.platform_index : '--';
                row.insertCell().textContent = p.platform_name || '--';
                row.insertCell().textContent = p.sensors || '无';
                row.insertCell().textContent = p.platform_type || '--';
                row.insertCell().textContent = p.side || '--';
            });
        } else {
            const row = tbody.insertRow();
            const cell = row.insertCell();
            cell.colSpan = 5;
            cell.textContent = '暂无平台数据';
            cell.style.textAlign = 'center';
        }

        container.appendChild(table);
    } catch (error) {
        console.error('处理平台信息数据时出错:', error);
        if (container) container.innerHTML = '<p style="color:red;text-align:center;">无法处理平台信息。</p>';
    }
}

// ====================================================================================
// jQuery DOM Ready
// ====================================================================================
$(function () {
    let currentPlatformData = null;
    let currentSimulationData = null;
    let lastPlatformDataString = "";
    let lastSimulationDataString = "";

    const REFRESH_INTERVAL = 5000;

    // ====================================================================================
    // 协同效能统计
    // ====================================================================================
    let simTargets = [];
    let simSelectedIndex = -1;

    function renderSimTargetList() {
        const listEl = document.getElementById('sim_target_list');
        if (!listEl) return;

        if (!simTargets || simTargets.length === 0) {
            listEl.innerHTML = '<span class="sim-no-target">暂无探测目标</span>';
            renderSimDetail(-1);
            return;
        }

        listEl.innerHTML = '';
        simTargets.forEach(function(t, i) {
            const btn = document.createElement('div');
            btn.className = 'sim-target-btn' + (i === simSelectedIndex ? ' active' : '');
            btn.textContent = t.node_id || ('目标' + i);
            btn.setAttribute('data-sim-idx', String(i));
            btn.style.display = 'inline-block';
            btn.onclick = function() {
                simSelectedIndex = i;
                renderSimDetail(i);
                var allBtns = listEl.querySelectorAll('.sim-target-btn');
                allBtns.forEach(function(b) {
                    var bIdx = parseInt(b.getAttribute('data-sim-idx'));
                    if (bIdx === i) b.classList.add('active');
                    else b.classList.remove('active');
                });
            };
            listEl.appendChild(btn);
        });

        if (simSelectedIndex < 0 || simSelectedIndex >= simTargets.length) {
            simSelectedIndex = 0;
        }
        renderSimDetail(simSelectedIndex);
    }

    function renderSimDetail(index) {
        const nodeEl = document.getElementById('sim_node_id');
        const discEl = document.getElementById('sim_discovery_time');
        const oodaEl = document.getElementById('sim_ooda_time');
        if (!nodeEl || !discEl || !oodaEl) return;

        if (index < 0 || !simTargets || index >= simTargets.length) {
            nodeEl.textContent = '--';
            discEl.textContent = '--';
            oodaEl.textContent = '--';
            return;
        }

        const t = simTargets[index];
        nodeEl.textContent = t.node_id || '--';
        discEl.textContent = t.target_discovery_time || '--';
        oodaEl.textContent = t.ooda_loop_time || '--';
    }

    function updateSimulationStats(simData) {
        if (simData && simData.targets && Array.isArray(simData.targets)) {
            simTargets = simData.targets;
        } else {
            simTargets = [];
        }
        if (simSelectedIndex >= simTargets.length) {
            simSelectedIndex = Math.max(0, simTargets.length - 1);
        }
        renderSimTargetList();
    }

    // ====================================================================================
    // 轮询
    // ====================================================================================
    function checkAndUpdateSimulationStats() {
        fetch('simulation_stats.json?t=' + new Date().getTime())
            .then(response => { if (!response.ok) throw new Error('fetch failed'); return response.json(); })
            .then(newData => {
                const s = JSON.stringify(newData);
                if (s !== lastSimulationDataString) {
                    currentSimulationData = newData;
                    lastSimulationDataString = s;
                    updateSimulationStats(currentSimulationData);
                }
            })
            .catch(error => { console.error('轮询 simulation_stats.json 失败:', error); });
    }

    function checkAndUpdatePlatformInfo() {
        fetch('platform_info.json?t=' + new Date().getTime())
            .then(response => { if (!response.ok) throw new Error('fetch failed'); return response.json(); })
            .then(newData => {
                const s = JSON.stringify(newData);
                if (s !== lastPlatformDataString) {
                    currentPlatformData = newData;
                    lastPlatformDataString = s;
                    loadPlatformInfo(currentPlatformData);
                }
            })
            .catch(error => { console.error('轮询 platform_info.json 失败:', error); });
    }

    // ====================================================================================
    // 初始化
    // ====================================================================================
    function initialLoad() {
        fetch('simulation_stats.json?t=' + new Date().getTime())
            .then(r => { if (!r.ok) throw new Error('fetch failed'); return r.json(); })
            .then(data => { currentSimulationData = data; lastSimulationDataString = JSON.stringify(data); updateSimulationStats(data); })
            .catch(e => { console.error('simulation_stats.json 加载失败:', e); updateSimulationStats(null); });

        fetch('platform_info.json?t=' + new Date().getTime())
            .then(r => { if (!r.ok) throw new Error('fetch failed'); return r.json(); })
            .then(data => { currentPlatformData = data; lastPlatformDataString = JSON.stringify(data); loadPlatformInfo(data); })
            .catch(e => { console.error('platform_info.json 加载失败:', e); loadPlatformInfo(null); });

        setInterval(checkAndUpdateSimulationStats, REFRESH_INTERVAL);
        setInterval(checkAndUpdatePlatformInfo, REFRESH_INTERVAL);
    }

    initialLoad();

}); // jQuery ready end
