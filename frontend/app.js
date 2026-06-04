const API_BASE = '/api';
const COLORS = {
    cql: '#ff6b6b',
    cql_rnn: '#ff9f43',
    dqn: '#ffd93d',
    behavior_cloning: '#6bcf7f',
};
const LABELS = {
    cql: 'CQL',
    cql_rnn: 'CQL+RNN',
    dqn: 'DQN',
    behavior_cloning: 'Behavior Cloning',
};

let charts = {};
let pollingIntervals = {};
let runData = {};

function initCharts() {
    const defaultOpts = {
        responsive: true,
        animation: { duration: 300 },
        plugins: { legend: { labels: { color: '#aaa' } } },
        scales: {
            x: { title: { display: true, text: 'Epoch', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
            y: { ticks: { color: '#666' }, grid: { color: '#333' } },
        },
    };

    charts.loss = new Chart(document.getElementById('chart-loss'), {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { ...defaultOpts, scales: { ...defaultOpts.scales, y: { ...defaultOpts.scales.y, title: { display: true, text: 'Loss', color: '#888' } } } },
    });

    charts.reward = new Chart(document.getElementById('chart-reward'), {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { ...defaultOpts, scales: { ...defaultOpts.scales, y: { ...defaultOpts.scales.y, title: { display: true, text: 'Cumulative Reward', color: '#888' } } } },
    });

    charts.qvalue = new Chart(document.getElementById('chart-qvalue'), {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { ...defaultOpts, scales: { ...defaultOpts.scales, y: { ...defaultOpts.scales.y, title: { display: true, text: 'Q-value Mean', color: '#888' } } } },
    });

    charts.penalty = new Chart(document.getElementById('chart-penalty'), {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { ...defaultOpts, scales: { ...defaultOpts.scales, y: { ...defaultOpts.scales.y, title: { display: true, text: 'CQL Penalty', color: '#888' } } } },
    });
}

async function generateData() {
    const btn = document.getElementById('btn-generate');
    btn.disabled = true;
    btn.textContent = '生成中...';

    try {
        await fetch(`${API_BASE}/data/generate`, { method: 'POST' });
        pollDataProgress();
    } catch (e) {
        btn.disabled = false;
        btn.textContent = '生成100万离线数据';
        alert('生成失败: ' + e.message);
    }
}

function pollDataProgress() {
    const interval = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/data/status`);
            const data = await res.json();
            const progressBar = document.getElementById('data-progress');
            const statusEl = document.getElementById('data-status');

            progressBar.style.width = (data.progress * 100) + '%';
            statusEl.textContent = `${(data.progress * 100).toFixed(1)}% (${data.total_generated.toLocaleString()} / 1,000,000)`;

            if (!data.is_running && data.progress >= 1.0) {
                clearInterval(interval);
                const btn = document.getElementById('btn-generate');
                btn.disabled = false;
                btn.textContent = '数据已就绪';
                statusEl.textContent = '完成: 1,000,000 条数据';
            }
        } catch (e) {
            console.error(e);
        }
    }, 2000);
}

async function startTraining(algorithm) {
    const hyperparameters = {
        alpha: parseFloat(document.getElementById('param-alpha').value),
        lr: parseFloat(document.getElementById('param-lr').value),
        epochs: parseInt(document.getElementById('param-epochs').value),
    };

    try {
        const res = await fetch(`${API_BASE}/training/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ algorithm, hyperparameters }),
        });
        const run = await res.json();
        pollTrainingMetrics(run.id, algorithm);
        refreshRuns();
    } catch (e) {
        alert('启动训练失败: ' + e.message);
    }
}

function pollTrainingMetrics(runId, algorithm) {
    if (pollingIntervals[runId]) return;

    runData[runId] = { algorithm, epochs: [], losses: [], rewards: [], qvalues: [], penalties: [] };

    pollingIntervals[runId] = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/metrics/latest/${runId}`);
            if (!res.ok) return;
            const data = await res.json();

            if (data.status === 'idle') {
                clearInterval(pollingIntervals[runId]);
                delete pollingIntervals[runId];
                await loadFullMetrics(runId, algorithm);
                refreshRuns();
                return;
            }

            const rd = runData[runId];
            const epoch = data.epoch;
            if (rd.epochs.length === 0 || rd.epochs[rd.epochs.length - 1] < epoch) {
                rd.epochs.push(epoch);
                rd.losses.push(data.metrics.loss);
                rd.rewards.push(data.cumulative_reward);
                rd.qvalues.push(data.metrics.q_value_mean);
                rd.penalties.push(data.metrics.cql_penalty || 0);
                updateCharts();
            }

            if (epoch >= data.total_epochs) {
                clearInterval(pollingIntervals[runId]);
                delete pollingIntervals[runId];
                refreshRuns();
            }
        } catch (e) {
            console.error(e);
        }
    }, 2000);
}

async function loadFullMetrics(runId, algorithm) {
    try {
        const res = await fetch(`${API_BASE}/metrics/runs/${runId}`);
        const data = await res.json();

        runData[runId] = {
            algorithm: data.algorithm,
            epochs: data.metrics.map(m => m.epoch),
            losses: data.metrics.map(m => m.loss),
            rewards: data.metrics.map(m => m.cumulative_reward),
            qvalues: data.metrics.map(m => m.q_value_mean || 0),
            penalties: data.metrics.map(m => m.cql_penalty || 0),
        };
        updateCharts();
    } catch (e) {
        console.error(e);
    }
}

function updateCharts() {
    const runIds = Object.keys(runData);

    function buildDatasets(field) {
        return runIds.map(rid => {
            const rd = runData[rid];
            return {
                label: `${LABELS[rd.algorithm] || rd.algorithm} (#${rid})`,
                data: rd[field],
                borderColor: COLORS[rd.algorithm] || '#fff',
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
            };
        });
    }

    const maxEpochs = Math.max(...runIds.map(rid => runData[rid].epochs.length), 0);
    const labels = Array.from({ length: maxEpochs }, (_, i) => i + 1);

    charts.loss.data = { labels, datasets: buildDatasets('losses') };
    charts.loss.update('none');

    charts.reward.data = { labels, datasets: buildDatasets('rewards') };
    charts.reward.update('none');

    charts.qvalue.data = { labels, datasets: buildDatasets('qvalues') };
    charts.qvalue.update('none');

    const penaltyDatasets = runIds
        .filter(rid => runData[rid].algorithm === 'cql')
        .map(rid => {
            const rd = runData[rid];
            return {
                label: `CQL Penalty (#${rid})`,
                data: rd.penalties,
                borderColor: COLORS.cql,
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
            };
        });
    charts.penalty.data = { labels, datasets: penaltyDatasets };
    charts.penalty.update('none');
}

async function refreshRuns() {
    try {
        const res = await fetch(`${API_BASE}/training/runs`);
        const runs = await res.json();
        const tbody = document.getElementById('runs-tbody');
        tbody.innerHTML = runs.map(r => `
            <tr>
                <td>${r.id}</td>
                <td style="color:${COLORS[r.algorithm] || '#fff'}">${LABELS[r.algorithm] || r.algorithm}</td>
                <td>${r.status}</td>
                <td>${r.total_epochs}</td>
                <td>${r.best_reward ? r.best_reward.toFixed(2) : '-'}</td>
                <td><button onclick="loadFullMetrics(${r.id}, '${r.algorithm}')">加载曲线</button></td>
            </tr>
        `).join('');

        const select = document.getElementById('q-dist-run-select');
        const completedRuns = runs.filter(r => r.status === 'completed' && (r.algorithm === 'cql' || r.algorithm === 'dqn' || r.algorithm === 'cql_rnn'));
        select.innerHTML = '<option value="">-- 选择 --</option>' +
            completedRuns.map(r => `<option value="${r.id}">${LABELS[r.algorithm] || r.algorithm} #${r.id} (奖励: ${r.best_reward ? r.best_reward.toFixed(2) : '-'})</option>`).join('');
    } catch (e) {
        console.error(e);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initQDistChart();
    refreshRuns();
    loadShiftRecords();
    loadAlerts();
});

// ===== Q-Value Distribution Histogram =====
let qDistChart = null;

function initQDistChart() {
    const ctx = document.getElementById('chart-q-distribution');
    qDistChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            animation: { duration: 300 },
            plugins: {
                legend: { labels: { color: '#aaa' } },
                title: { display: true, text: 'Q-values per Action (蓝=In-Distribution, 红=OOD)', color: '#aaa' },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Action Index', color: '#888' },
                    ticks: { color: '#666', maxTicksLimit: 20 },
                    grid: { color: '#333' },
                },
                y: {
                    title: { display: true, text: 'Q-value', color: '#888' },
                    ticks: { color: '#666' },
                    grid: { color: '#333' },
                },
            },
        },
    });
}

async function loadQDistribution() {
    const select = document.getElementById('q-dist-run-select');
    const runId = select.value;
    if (!runId) return;

    try {
        const res = await fetch(`${API_BASE}/shift/q_distribution/${runId}`);
        if (!res.ok) {
            console.error('Failed to load Q distribution');
            return;
        }
        const data = await res.json();

        const inDistColors = [];
        const oodColors = [];
        const bgColors = [];

        for (let i = 0; i < data.action_indices.length; i++) {
            if (data.in_distribution_mask[i]) {
                bgColors.push('rgba(54, 162, 235, 0.7)');
            } else {
                bgColors.push('rgba(255, 99, 132, 0.7)');
            }
        }

        const inDistQ = data.q_values.filter((_, i) => data.in_distribution_mask[i]);
        const oodQ = data.q_values.filter((_, i) => !data.in_distribution_mask[i]);
        const inDistMean = inDistQ.reduce((a, b) => a + b, 0) / (inDistQ.length || 1);
        const oodMean = oodQ.reduce((a, b) => a + b, 0) / (oodQ.length || 1);

        qDistChart.data = {
            labels: data.action_indices,
            datasets: [{
                label: `Q-values (In-Dist均值=${inDistMean.toFixed(3)}, OOD均值=${oodMean.toFixed(3)})`,
                data: data.q_values,
                backgroundColor: bgColors,
                borderWidth: 0,
            }],
        };
        qDistChart.update();
    } catch (e) {
        console.error('Error loading Q distribution:', e);
    }
}

// ===== Shift Detection =====
async function runShiftDetection() {
    try {
        const res = await fetch(`${API_BASE}/shift/detect`, { method: 'POST' });
        const results = await res.json();
        displayShiftResults(results);
        loadShiftRecords();
        loadAlerts();
    } catch (e) {
        alert('偏移检测失败: ' + e.message);
    }
}

async function runShiftWithNewItems() {
    try {
        const res = await fetch(`${API_BASE}/shift/detect_with_new_items?new_item_count=20`, { method: 'POST' });
        const results = await res.json();
        displayShiftResults(results);
        loadShiftRecords();
        loadAlerts();
    } catch (e) {
        alert('偏移检测失败: ' + e.message);
    }
}

function displayShiftResults(results) {
    const hasAlerts = results.some(r => r.is_alert);
    const panel = document.getElementById('shift-alerts');
    const container = document.getElementById('alerts-container');

    if (hasAlerts) {
        panel.style.display = 'block';
        container.innerHTML = results
            .filter(r => r.is_alert)
            .map(r => `
                <div class="alert-item">
                    <span class="alert-type">${r.shift_type}</span>
                    <span class="alert-metric">${r.metric_name}: ${r.metric_value.toFixed(4)}</span>
                    <span class="alert-threshold">阈值: ${r.threshold.toFixed(4)}</span>
                </div>
            `).join('');
    } else {
        panel.style.display = 'none';
    }
}

async function loadAlerts() {
    try {
        const res = await fetch(`${API_BASE}/shift/alerts`);
        const alerts = await res.json();
        const panel = document.getElementById('shift-alerts');
        const container = document.getElementById('alerts-container');

        if (alerts.length > 0) {
            panel.style.display = 'block';
            container.innerHTML = alerts.slice(0, 5).map(a => `
                <div class="alert-item">
                    <span class="alert-type">${a.shift_type}</span>
                    <span class="alert-metric">${a.metric_name}: ${a.metric_value.toFixed(4)}</span>
                    <span class="alert-threshold">阈值: ${a.threshold.toFixed(4)}</span>
                    ${a.triggered_retrain ? '<span class="alert-retrain">已触发重训 #' + a.retrain_run_id + '</span>' : ''}
                </div>
            `).join('');
        }
    } catch (e) {
        console.error(e);
    }
}

async function loadShiftRecords() {
    try {
        const res = await fetch(`${API_BASE}/shift/records?limit=20`);
        const records = await res.json();
        const tbody = document.getElementById('shift-tbody');
        tbody.innerHTML = records.map(r => `
            <tr class="${r.is_alert ? 'row-alert' : ''}">
                <td>${r.detection_time ? new Date(r.detection_time).toLocaleString() : '-'}</td>
                <td>${r.shift_type}</td>
                <td>${r.metric_name}</td>
                <td>${r.metric_value.toFixed(4)}</td>
                <td>${r.threshold.toFixed(4)}</td>
                <td>${r.is_alert ? '<span class="badge-alert">告警</span>' : '<span class="badge-ok">正常</span>'}</td>
                <td>${r.triggered_retrain ? '是 (#' + r.retrain_run_id + ')' : '否'}</td>
            </tr>
        `).join('');
    } catch (e) {
        console.error(e);
    }
}
