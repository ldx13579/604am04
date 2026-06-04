const API_BASE = '/api';
const COLORS = {
    cql: '#ff6b6b',
    cql_rnn: '#ff9f43',
    dqn: '#ffd93d',
    behavior_cloning: '#6bcf7f',
    ensemble_cql: '#a55eea',
};
const LABELS = {
    cql: 'CQL',
    cql_rnn: 'CQL+RNN',
    dqn: 'DQN',
    behavior_cloning: 'Behavior Cloning',
    ensemble_cql: 'Ensemble CQL',
};

let charts = {};
let pollingIntervals = {};
let runData = {};
let smoothingFactor = 0;
let hiddenRuns = new Set();

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

    if (algorithm === 'ensemble_cql') {
        hyperparameters.uncertainty_threshold = parseFloat(document.getElementById('param-uncertainty').value);
    }

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
    updateRunToggles();

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
        updateRunToggles();

        if (data.algorithm === 'ensemble_cql') {
            loadEnsembleUncertainty(runId);
            loadExplorationRatio(runId);
            loadPerModelLosses(runId);
        }
    } catch (e) {
        console.error(e);
    }
}

function smoothData(data, factor) {
    if (factor <= 0 || data.length === 0) return data;
    const result = [data[0]];
    for (let i = 1; i < data.length; i++) {
        result.push(factor * result[i - 1] + (1 - factor) * data[i]);
    }
    return result;
}

function updateCharts() {
    const runIds = Object.keys(runData).filter(rid => !hiddenRuns.has(rid));

    function buildDatasets(field) {
        return runIds.map(rid => {
            const rd = runData[rid];
            return {
                label: `${LABELS[rd.algorithm] || rd.algorithm} (#${rid})`,
                data: smoothData(rd[field], smoothingFactor),
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
        .filter(rid => ['cql', 'cql_rnn', 'ensemble_cql'].includes(runData[rid].algorithm))
        .map(rid => {
            const rd = runData[rid];
            return {
                label: `${LABELS[rd.algorithm] || rd.algorithm} Penalty (#${rid})`,
                data: smoothData(rd.penalties, smoothingFactor),
                borderColor: COLORS[rd.algorithm] || COLORS.cql,
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
        const completedRuns = runs.filter(r => r.status === 'completed' && ['cql', 'dqn', 'cql_rnn', 'ensemble_cql'].includes(r.algorithm));
        select.innerHTML = '<option value="">-- 选择 --</option>' +
            completedRuns.map(r => `<option value="${r.id}">${LABELS[r.algorithm] || r.algorithm} #${r.id} (奖励: ${r.best_reward ? r.best_reward.toFixed(2) : '-'})</option>`).join('');
    } catch (e) {
        console.error(e);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    initUncertaintyChart();
    initFQEChart();
    initFQELossChart();
    initExplorationChart();
    initPerModelChart();
    initAlphaComparisonChart();
    initQDistChart();
    initQHistogramChart();
    initShiftTimelineChart();
    initToolbarControls();
    refreshRuns();
    loadShiftRecords();
    loadAlerts();
    loadPolicyVersions();
});

// ===== Ensemble Uncertainty Chart =====
let uncertaintyChart = null;

function initUncertaintyChart() {
    const ctx = document.getElementById('chart-uncertainty');
    uncertaintyChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            animation: { duration: 300 },
            plugins: { legend: { labels: { color: '#aaa' } } },
            scales: {
                x: { title: { display: true, text: 'Epoch', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
                y: { title: { display: true, text: 'Q-value Std Dev', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
            },
        },
    });
}

async function loadEnsembleUncertainty(runId) {
    try {
        const res = await fetch(`${API_BASE}/evaluation/ensemble/uncertainty/${runId}`);
        if (!res.ok) return;
        const data = await res.json();

        uncertaintyChart.data = {
            labels: data.map(m => m.epoch),
            datasets: [
                {
                    label: `Uncertainty Mean (#${runId})`,
                    data: data.map(m => m.uncertainty_mean),
                    borderColor: '#a55eea',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                },
                {
                    label: `Uncertainty Max (#${runId})`,
                    data: data.map(m => m.uncertainty_max),
                    borderColor: '#ff6b6b',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    pointRadius: 0,
                    tension: 0.3,
                },
            ],
        };
        uncertaintyChart.update();
    } catch (e) {
        console.error('Error loading ensemble uncertainty:', e);
    }
}

// ===== FQE Estimated Value Chart =====
let fqeChart = null;
let fqePollingIntervals = {};

function initFQEChart() {
    const ctx = document.getElementById('chart-fqe-value');
    fqeChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            animation: { duration: 300 },
            plugins: { legend: { labels: { color: '#aaa' } } },
            scales: {
                x: { title: { display: true, text: 'FQE Epoch', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
                y: { title: { display: true, text: 'Estimated Policy Value', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
            },
        },
    });
}

async function startFQE() {
    const sourceRunId = parseInt(document.getElementById('fqe-source-run').value);
    if (!sourceRunId) {
        alert('请输入有效的源Run ID');
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/evaluation/fqe/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ source_run_id: sourceRunId }),
        });
        if (!res.ok) {
            const err = await res.json();
            alert('FQE启动失败: ' + (err.detail || '未知错误'));
            return;
        }
        const evaluation = await res.json();
        pollFQEProgress(evaluation.id);
    } catch (e) {
        alert('FQE启动失败: ' + e.message);
    }
}

function pollFQEProgress(evaluationId) {
    if (fqePollingIntervals[evaluationId]) return;

    fqePollingIntervals[evaluationId] = setInterval(async () => {
        try {
            const res = await fetch(`${API_BASE}/evaluation/fqe/latest/${evaluationId}`);
            if (!res.ok) return;
            const data = await res.json();

            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(fqePollingIntervals[evaluationId]);
                delete fqePollingIntervals[evaluationId];
                await loadFQEResults(evaluationId);
                return;
            }

            if (data.epoch && data.estimated_value !== null) {
                fqeChart.data.labels.push(data.epoch);
                fqeChart.data.datasets = [{
                    label: `FQE Value (#${evaluationId})`,
                    data: [...(fqeChart.data.datasets[0]?.data || []), data.estimated_value],
                    borderColor: '#45b7d1',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    pointRadius: 0,
                    tension: 0.3,
                }];
                fqeChart.update('none');
            }
        } catch (e) {
            console.error(e);
        }
    }, 2000);
}

async function loadFQEResults(evaluationId) {
    try {
        const res = await fetch(`${API_BASE}/evaluation/fqe/results/${evaluationId}`);
        if (!res.ok) return;
        const data = await res.json();

        fqeChart.data = {
            labels: data.metrics.map(m => m.epoch),
            datasets: [{
                label: `FQE Estimated Value (#${evaluationId})`,
                data: data.metrics.map(m => m.estimated_value),
                borderColor: '#45b7d1',
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
            }],
        };
        fqeChart.update();
        loadFQELoss(evaluationId);
    } catch (e) {
        console.error('Error loading FQE results:', e);
    }
}

// ===== Exploration Ratio Chart =====
let explorationChart = null;

function initExplorationChart() {
    const ctx = document.getElementById('chart-exploration');
    explorationChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            animation: { duration: 300 },
            plugins: { legend: { labels: { color: '#aaa' } } },
            scales: {
                x: { title: { display: true, text: 'Epoch', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
                y: { title: { display: true, text: 'Exploration Ratio', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' }, min: 0, max: 1 },
            },
        },
    });
}

async function loadExplorationRatio(runId) {
    try {
        const res = await fetch(`${API_BASE}/evaluation/ensemble/uncertainty/${runId}`);
        if (!res.ok) return;
        const data = await res.json();

        explorationChart.data = {
            labels: data.map(m => m.epoch),
            datasets: [{
                label: `Exploration Ratio (#${runId})`,
                data: data.map(m => m.exploration_ratio),
                borderColor: '#ff9f43',
                backgroundColor: 'rgba(255, 159, 67, 0.1)',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
                fill: true,
            }],
        };
        explorationChart.update();
    } catch (e) {
        console.error('Error loading exploration ratio:', e);
    }
}

// ===== Alpha Comparison Chart =====
let alphaComparisonChart = null;

function initAlphaComparisonChart() {
    const ctx = document.getElementById('chart-alpha-comparison');
    alphaComparisonChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            animation: { duration: 300 },
            plugins: { legend: { labels: { color: '#aaa' } } },
            scales: {
                x: { title: { display: true, text: 'Alpha Value', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
                y: { title: { display: true, text: 'Final Reward', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' }, position: 'left' },
                y2: { title: { display: true, text: 'Convergence Epoch', color: '#888' }, ticks: { color: '#666' }, grid: { display: false }, position: 'right' },
            },
        },
    });
}

async function startAlphaSweep() {
    const input = document.getElementById('alpha-sweep-values').value;
    const alphaValues = input.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
    if (alphaValues.length === 0) {
        alert('请输入有效的Alpha值');
        return;
    }

    try {
        const res = await fetch(`${API_BASE}/evaluation/hyperparams/alpha_sweep`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ alpha_values: alphaValues }),
        });
        if (!res.ok) {
            const err = await res.json();
            alert('Alpha Sweep启动失败: ' + (err.detail || '未知错误'));
            return;
        }
        const data = await res.json();
        alert(`已启动 ${data.run_ids.length} 个训练任务 (Run IDs: ${data.run_ids.join(', ')})`);
        refreshRuns();
    } catch (e) {
        alert('Alpha Sweep启动失败: ' + e.message);
    }
}

async function loadAlphaComparison() {
    try {
        const runsRes = await fetch(`${API_BASE}/training/runs`);
        const allRuns = await runsRes.json();
        const cqlRuns = allRuns.filter(r => r.algorithm === 'cql' && r.status === 'completed');

        if (cqlRuns.length === 0) {
            alert('没有已完成的CQL训练记录');
            return;
        }

        const runIds = cqlRuns.map(r => r.id).join(',');
        const res = await fetch(`${API_BASE}/evaluation/hyperparams/alpha_comparison?run_ids=${runIds}`);
        if (!res.ok) return;
        const data = await res.json();

        const comparisons = data.comparisons.sort((a, b) => a.alpha - b.alpha);

        alphaComparisonChart.data = {
            labels: comparisons.map(c => `α=${c.alpha}`),
            datasets: [
                {
                    label: 'Final Reward',
                    data: comparisons.map(c => c.final_reward || 0),
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1,
                    yAxisID: 'y',
                },
                {
                    label: 'Convergence Epoch',
                    data: comparisons.map(c => c.convergence_epoch || 0),
                    backgroundColor: 'rgba(255, 159, 67, 0.6)',
                    borderColor: 'rgba(255, 159, 67, 1)',
                    borderWidth: 1,
                    yAxisID: 'y2',
                },
            ],
        };
        alphaComparisonChart.update();
    } catch (e) {
        console.error('Error loading alpha comparison:', e);
    }
}

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
        tbody.innerHTML = records.map(r => {
            const severity = r.is_alert ? (r.metric_value >= r.threshold * 2 ? 'severe' : 'warning') : 'normal';
            const severityBadge = severity === 'severe'
                ? '<span class="badge-severe">严重</span>'
                : severity === 'warning'
                    ? '<span class="badge-warning">警告</span>'
                    : '<span class="badge-ok">正常</span>';
            return `
            <tr class="${r.is_alert ? 'row-alert' : ''}">
                <td>${r.detection_time ? new Date(r.detection_time).toLocaleString() : '-'}</td>
                <td>${r.shift_type}</td>
                <td>${r.metric_name}</td>
                <td>${r.metric_value.toFixed(4)}</td>
                <td>${r.threshold.toFixed(4)}</td>
                <td>${severityBadge}</td>
                <td>${r.is_alert ? '<span class="badge-alert">告警</span>' : '<span class="badge-ok">正常</span>'}</td>
                <td>${r.triggered_retrain ? '<span class="badge-retrain">是 (#' + r.retrain_run_id + ')</span>' : '否'}</td>
            </tr>`;
        }).join('');

        updateShiftTimeline(records);
    } catch (e) {
        console.error(e);
    }
}

// ===== Q-Value Binned Histogram =====
let qHistogramChart = null;

function initQHistogramChart() {
    const ctx = document.getElementById('chart-q-histogram');
    qHistogramChart = new Chart(ctx, {
        type: 'bar',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            animation: { duration: 300 },
            plugins: {
                legend: { labels: { color: '#aaa' } },
                title: { display: true, text: 'Q值分布: In-Distribution (蓝) vs OOD (红)', color: '#aaa' },
            },
            scales: {
                x: {
                    title: { display: true, text: 'Q-value Range', color: '#888' },
                    ticks: { color: '#666', maxTicksLimit: 15 },
                    grid: { color: '#333' },
                },
                y: {
                    title: { display: true, text: 'Count', color: '#888' },
                    ticks: { color: '#666' },
                    grid: { color: '#333' },
                },
            },
        },
    });
}

async function loadQHistogram() {
    const select = document.getElementById('q-dist-run-select');
    const runId = select.value;
    if (!runId) return;

    try {
        const res = await fetch(`${API_BASE}/shift/q_histogram/${runId}?n_states=50&n_bins=30`);
        if (!res.ok) {
            console.error('Failed to load Q histogram');
            return;
        }
        const data = await res.json();

        const binLabels = [];
        for (let i = 0; i < data.bin_edges.length - 1; i++) {
            binLabels.push(((data.bin_edges[i] + data.bin_edges[i + 1]) / 2).toFixed(3));
        }

        qHistogramChart.data = {
            labels: binLabels,
            datasets: [
                {
                    label: `In-Distribution (均值=${data.in_dist_mean.toFixed(4)})`,
                    data: data.in_dist_counts,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1,
                },
                {
                    label: `OOD (均值=${data.ood_mean.toFixed(4)})`,
                    data: data.ood_counts,
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1,
                },
            ],
        };
        qHistogramChart.update();

        const statsPanel = document.getElementById('q-hist-stats');
        statsPanel.style.display = 'flex';
        document.getElementById('stat-in-dist-mean').textContent = `均值: ${data.in_dist_mean.toFixed(4)}`;
        document.getElementById('stat-in-dist-std').textContent = `标准差: ${data.in_dist_std.toFixed(4)}`;
        document.getElementById('stat-ood-mean').textContent = `均值: ${data.ood_mean.toFixed(4)}`;
        document.getElementById('stat-ood-std').textContent = `标准差: ${data.ood_std.toFixed(4)}`;
        const gap = data.in_dist_mean - data.ood_mean;
        const pooledStd = Math.sqrt((data.in_dist_std ** 2 + data.ood_std ** 2) / 2);
        const effectSize = pooledStd > 0 ? gap / pooledStd : 0;
        document.getElementById('stat-gap').textContent = `差距: ${gap.toFixed(4)}`;
        document.getElementById('stat-effect-size').textContent = `效应量(Cohen's d): ${effectSize.toFixed(3)}`;
    } catch (e) {
        console.error('Error loading Q histogram:', e);
    }
}

// ===== Shift Detection Timeline Chart =====
let shiftTimelineChart = null;

function initShiftTimelineChart() {
    const ctx = document.getElementById('chart-shift-timeline');
    shiftTimelineChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            animation: { duration: 300 },
            plugins: {
                legend: { labels: { color: '#aaa' } },
                title: { display: true, text: '偏移指标时间线 (超过阈值线=告警)', color: '#aaa' },
            },
            scales: {
                x: {
                    title: { display: true, text: '检测时间', color: '#888' },
                    ticks: { color: '#666', maxTicksLimit: 10 },
                    grid: { color: '#333' },
                },
                y: {
                    title: { display: true, text: '指标值 / 阈值比', color: '#888' },
                    ticks: { color: '#666' },
                    grid: { color: '#333' },
                },
            },
        },
    });
}

function updateShiftTimeline(records) {
    if (!records || records.length === 0) return;

    const sorted = [...records].reverse();
    const shiftTypes = [...new Set(sorted.map(r => r.shift_type))];
    const typeColors = {
        action_distribution: '#ff6b6b',
        reward_distribution: '#ffd93d',
        state_distribution: '#6bcf7f',
        new_items: '#a55eea',
    };

    const labels = sorted.map(r => r.detection_time ? new Date(r.detection_time).toLocaleTimeString() : '-');
    const datasets = shiftTypes.map(type => {
        const data = sorted.map(r => r.shift_type === type ? r.metric_value / r.threshold : null);
        return {
            label: type,
            data,
            borderColor: typeColors[type] || '#fff',
            backgroundColor: 'transparent',
            borderWidth: 2,
            pointRadius: 3,
            tension: 0.3,
            spanGaps: true,
        };
    });

    datasets.push({
        label: '告警阈值',
        data: sorted.map(() => 1.0),
        borderColor: 'rgba(255, 99, 132, 0.5)',
        borderDash: [5, 5],
        borderWidth: 2,
        pointRadius: 0,
        backgroundColor: 'transparent',
    });

    shiftTimelineChart.data = { labels, datasets };
    shiftTimelineChart.update();
}

// ===== TensorBoard Toolbar Controls =====
function initToolbarControls() {
    const smoothingSlider = document.getElementById('tb-smoothing');
    const smoothingDisplay = document.getElementById('tb-smoothing-value');
    smoothingSlider.addEventListener('input', () => {
        smoothingFactor = parseFloat(smoothingSlider.value);
        smoothingDisplay.textContent = smoothingFactor.toFixed(2);
        updateCharts();
    });

    const logScaleCheckbox = document.getElementById('tb-logscale');
    logScaleCheckbox.addEventListener('change', () => {
        charts.loss.options.scales.y.type = logScaleCheckbox.checked ? 'logarithmic' : 'linear';
        charts.loss.update();
    });
}

function updateRunToggles() {
    const container = document.getElementById('tb-run-toggles');
    const allRunIds = Object.keys(runData);
    container.innerHTML = allRunIds.map(rid => {
        const rd = runData[rid];
        const checked = !hiddenRuns.has(rid) ? 'checked' : '';
        const color = COLORS[rd.algorithm] || '#fff';
        return `<label class="tb-toggle-label" style="border-color:${color}">
            <input type="checkbox" ${checked} onchange="toggleRunVisibility('${rid}')">
            <span style="color:${color}">${LABELS[rd.algorithm] || rd.algorithm} #${rid}</span>
        </label>`;
    }).join('');
}

function toggleRunVisibility(runId) {
    if (hiddenRuns.has(runId)) {
        hiddenRuns.delete(runId);
    } else {
        hiddenRuns.add(runId);
    }
    updateCharts();
}

// ===== FQE Loss Chart =====
let fqeLossChart = null;

function initFQELossChart() {
    const ctx = document.getElementById('chart-fqe-loss');
    fqeLossChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            animation: { duration: 300 },
            plugins: { legend: { labels: { color: '#aaa' } } },
            scales: {
                x: { title: { display: true, text: 'FQE Epoch', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
                y: { title: { display: true, text: 'FQE Loss', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
            },
        },
    });
}

async function loadFQELoss(evaluationId) {
    try {
        const res = await fetch(`${API_BASE}/evaluation/fqe/results/${evaluationId}`);
        if (!res.ok) return;
        const data = await res.json();

        fqeLossChart.data = {
            labels: data.metrics.map(m => m.epoch),
            datasets: [{
                label: `FQE Loss (#${evaluationId})`,
                data: data.metrics.map(m => m.fqe_loss),
                borderColor: '#ff9f43',
                backgroundColor: 'transparent',
                borderWidth: 2,
                pointRadius: 0,
                tension: 0.3,
            }],
        };
        fqeLossChart.update();
    } catch (e) {
        console.error('Error loading FQE loss:', e);
    }
}

// ===== Per-Model Losses Chart =====
let perModelChart = null;

function initPerModelChart() {
    const ctx = document.getElementById('chart-per-model');
    perModelChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: {
            responsive: true,
            animation: { duration: 300 },
            plugins: { legend: { labels: { color: '#aaa' } } },
            scales: {
                x: { title: { display: true, text: 'Epoch', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
                y: { title: { display: true, text: 'Loss', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
            },
        },
    });
}

async function loadPerModelLosses(runId) {
    try {
        const res = await fetch(`${API_BASE}/evaluation/ensemble/uncertainty/${runId}`);
        if (!res.ok) return;
        const data = await res.json();

        const epochs = data.map(m => m.epoch);
        const modelColors = ['#ff6b6b', '#ff9f43', '#ffd93d', '#6bcf7f', '#45b7d1', '#a55eea', '#e056a0'];

        const validData = data.filter(m => m.per_model_losses && m.per_model_losses.length > 0);
        if (validData.length === 0) return;

        const nModels = validData[0].per_model_losses.length;
        const datasets = [];
        for (let i = 0; i < nModels; i++) {
            datasets.push({
                label: `Model ${i + 1}`,
                data: data.map(m => m.per_model_losses ? m.per_model_losses[i] : null),
                borderColor: modelColors[i % modelColors.length],
                backgroundColor: 'transparent',
                borderWidth: 1.5,
                pointRadius: 0,
                tension: 0.3,
            });
        }

        perModelChart.data = { labels: epochs, datasets };
        perModelChart.update();
    } catch (e) {
        console.error('Error loading per-model losses:', e);
    }
}

// ===== Policy Version Management =====
async function createPolicyVersion() {
    const runId = parseInt(document.getElementById('ver-run-id').value);
    const snapshotId = parseInt(document.getElementById('ver-snapshot-id').value);
    const versionTag = document.getElementById('ver-tag').value.trim();
    const fqeId = parseInt(document.getElementById('ver-fqe-id').value) || null;

    if (!runId || !snapshotId || !versionTag) {
        alert('请填写Run ID、Snapshot ID和版本标签');
        return;
    }

    try {
        const body = {
            run_id: runId,
            snapshot_id: snapshotId,
            version_tag: versionTag,
        };
        if (fqeId) body.fqe_evaluation_id = fqeId;

        const res = await fetch(`${API_BASE}/evaluation/versions`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!res.ok) {
            const err = await res.json();
            alert('创建版本失败: ' + (err.detail || '未知错误'));
            return;
        }
        loadPolicyVersions();
    } catch (e) {
        alert('创建版本失败: ' + e.message);
    }
}

async function loadPolicyVersions() {
    try {
        const res = await fetch(`${API_BASE}/evaluation/versions`);
        if (!res.ok) return;
        const versions = await res.json();
        const tbody = document.getElementById('versions-tbody');

        tbody.innerHTML = versions.map(v => {
            const stageBadge = v.stage === 'production'
                ? '<span class="badge-production">生产</span>'
                : v.stage === 'staging'
                    ? '<span class="badge-staging">预发布</span>'
                    : v.stage === 'archived'
                        ? '<span class="badge-archived">归档</span>'
                        : '<span class="badge-candidate">候选</span>';
            return `
            <tr>
                <td>${v.id}</td>
                <td>${v.version_tag}</td>
                <td>${stageBadge}</td>
                <td>Run #${v.run_id}</td>
                <td>${v.fqe_evaluation_id ? 'FQE #' + v.fqe_evaluation_id : '-'}</td>
                <td>
                    <button onclick="promoteVersion(${v.id}, 'staging')">预发布</button>
                    <button onclick="promoteVersion(${v.id}, 'production')">上线</button>
                    <button onclick="promoteVersion(${v.id}, 'archived')">归档</button>
                </td>
            </tr>`;
        }).join('');
    } catch (e) {
        console.error('Error loading policy versions:', e);
    }
}

async function promoteVersion(versionId, stage) {
    try {
        const res = await fetch(`${API_BASE}/evaluation/versions/${versionId}/stage`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ stage }),
        });
        if (!res.ok) {
            const err = await res.json();
            alert('更新阶段失败: ' + (err.detail || '未知错误'));
            return;
        }
        loadPolicyVersions();
    } catch (e) {
        alert('更新阶段失败: ' + e.message);
    }
}

async function loadVersionComparison() {
    try {
        const res = await fetch(`${API_BASE}/evaluation/versions`);
        if (!res.ok) return;
        const versions = await res.json();
        if (versions.length < 2) {
            alert('至少需要2个版本才能对比');
            return;
        }
        const ids = versions.slice(0, 5).map(v => v.id).join(',');
        const compRes = await fetch(`${API_BASE}/evaluation/versions/compare?ids=${ids}`);
        if (!compRes.ok) return;
        const data = await compRes.json();

        let html = '<table><thead><tr><th>版本</th><th>阶段</th><th>算法</th><th>最优奖励</th><th>FQE估值</th></tr></thead><tbody>';
        data.comparisons.forEach(c => {
            html += `<tr>
                <td>${c.version_tag}</td>
                <td>${c.stage}</td>
                <td>${c.algorithm || '-'}</td>
                <td>${c.best_reward ? c.best_reward.toFixed(2) : '-'}</td>
                <td>${c.fqe_estimated_value ? c.fqe_estimated_value.toFixed(4) : '-'}</td>
            </tr>`;
        });
        html += '</tbody></table>';

        const tbody = document.getElementById('versions-tbody');
        tbody.innerHTML = `<tr><td colspan="6">${html}</td></tr>` + tbody.innerHTML;
    } catch (e) {
        console.error('Error loading version comparison:', e);
    }
}


// =============================================
// Recommendation Service Simulator
// =============================================

async function simulateUser() {
    const state = Array.from({length: 10}, () => Math.random());
    const sum = state.reduce((a, b) => a + b, 0);
    const normalizedState = state.map(s => s / sum);

    try {
        const res = await fetch(`${API_BASE}/recommend/predict`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                user_state: normalizedState,
                top_k: 5,
                session_id: 'sim-' + Date.now()
            })
        });
        const data = await res.json();

        let html = '<div class="recommend-items">';
        data.items.forEach((item, i) => {
            const cat = Math.floor(item / 10);
            html += `<div class="recommend-item">
                <span>物品 #${item} (类别${cat})</span>
                <span class="score">Q=${data.scores[i].toFixed(3)}</span>
                <button class="btn-click" onclick="recordClick(${data.impression_id}, ${item}, true)">点击</button>
                <button class="btn-skip" onclick="recordClick(${data.impression_id}, ${item}, false)">跳过</button>
            </div>`;
        });
        html += '</div>';
        document.getElementById('recommend-result').innerHTML = html;

        const groupHtml = data.group
            ? `<span class="group-badge group-${data.group.toLowerCase()}">Group ${data.group} (${data.group === 'A' ? 'CQL' : 'Random'})</span>`
            : '<span class="group-badge">无实验</span>';
        document.getElementById('recommend-group').innerHTML = groupHtml;
    } catch (e) {
        console.error('Recommend error:', e);
    }
}

async function simulateBatch(n) {
    for (let i = 0; i < n; i++) {
        const state = Array.from({length: 10}, () => Math.random());
        const sum = state.reduce((a, b) => a + b, 0);
        const normalizedState = state.map(s => s / sum);

        try {
            const res = await fetch(`${API_BASE}/recommend/predict`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    user_state: normalizedState,
                    top_k: 5,
                    session_id: 'batch-' + Date.now() + '-' + i
                })
            });
            const data = await res.json();

            if (data.impression_id) {
                const clickItem = data.items[Math.floor(Math.random() * data.items.length)];
                const clicked = Math.random() < 0.3;
                await fetch(`${API_BASE}/recommend/feedback`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        impression_id: data.impression_id,
                        item_id: clickItem,
                        clicked: clicked
                    })
                });
            }
        } catch (e) {}
    }
    document.getElementById('recommend-result').innerHTML = `<div class="recommend-items"><span>完成 ${n} 次模拟请求</span></div>`;
    loadABSummary();
}

async function recordClick(impressionId, itemId, clicked) {
    if (!impressionId) return;
    try {
        await fetch(`${API_BASE}/recommend/feedback`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                impression_id: impressionId,
                item_id: itemId,
                clicked: clicked
            })
        });
    } catch (e) {}
}

async function loadPolicyInfo() {
    try {
        const res = await fetch(`${API_BASE}/recommend/policy_info`);
        const data = await res.json();
        const el = document.getElementById('policy-status');
        if (data.is_loaded) {
            el.innerHTML = `<span style="color:#6bcf7f">已加载</span> | 算法: ${data.algorithm || '-'} | 版本ID: ${data.policy_version_id || 'fallback'}`;
        } else {
            el.innerHTML = '<span style="color:#ff6b6b">未加载 (使用随机策略)</span>';
        }
    } catch (e) {}
}


// =============================================
// A/B Testing
// =============================================

let ctrChart = null;
let ctrPollingInterval = null;
let activeExperimentId = null;

function initCTRChart() {
    const ctx = document.getElementById('chart-ctr').getContext('2d');
    ctrChart = new Chart(ctx, {
        type: 'line',
        data: { labels: [], datasets: [
            { label: 'Group A (CQL)', data: [], borderColor: '#ff6b6b', fill: false, tension: 0.3 },
            { label: 'Group B (Random)', data: [], borderColor: '#ffd93d', fill: false, tension: 0.3 },
        ]},
        options: {
            responsive: true,
            plugins: { legend: { labels: { color: '#aaa' } } },
            scales: {
                x: { title: { display: true, text: '时间窗口', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
                y: { title: { display: true, text: 'CTR', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' }, min: 0, max: 1 },
            }
        }
    });
}

async function createABExperiment() {
    const name = document.getElementById('ab-name').value;
    const split = parseFloat(document.getElementById('ab-split').value);
    try {
        const res = await fetch(`${API_BASE}/ab/experiments`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name, traffic_split: split })
        });
        const data = await res.json();
        activeExperimentId = data.id;
        startCTRPolling();
        loadABExperiments();
    } catch (e) {
        console.error('Create AB error:', e);
    }
}

async function stopABExperiment() {
    if (!activeExperimentId) return;
    try {
        await fetch(`${API_BASE}/ab/experiments/${activeExperimentId}/status?status=completed`, { method: 'PUT' });
        if (ctrPollingInterval) clearInterval(ctrPollingInterval);
        loadABExperiments();
    } catch (e) {}
}

async function loadABExperiments() {
    try {
        const res = await fetch(`${API_BASE}/ab/experiments`);
        const experiments = await res.json();
        let html = '';
        experiments.forEach(exp => {
            const statusClass = exp.status === 'running' ? 'style="color:#6bcf7f"' : '';
            html += `<div><span ${statusClass}>[${exp.status}]</span> ${exp.name} (split: ${exp.traffic_split})
                     <button onclick="activeExperimentId=${exp.id};startCTRPolling();loadABSummary();">查看</button></div>`;
        });
        document.getElementById('ab-experiments-list').innerHTML = html || '无实验';
    } catch (e) {}
}

function startCTRPolling() {
    if (ctrPollingInterval) clearInterval(ctrPollingInterval);
    updateCTRChart();
    ctrPollingInterval = setInterval(updateCTRChart, 5000);
}

async function updateCTRChart() {
    if (!activeExperimentId) return;
    try {
        const res = await fetch(`${API_BASE}/ab/experiments/${activeExperimentId}/ctr?limit=50`);
        const data = await res.json();

        const labelsSet = [...new Set(data.map(d => d.window_start.slice(11, 19)))];
        const groupA = data.filter(d => d.group_name === 'A');
        const groupB = data.filter(d => d.group_name === 'B');

        ctrChart.data.labels = labelsSet;
        ctrChart.data.datasets[0].data = groupA.map(d => d.ctr);
        ctrChart.data.datasets[1].data = groupB.map(d => d.ctr);
        ctrChart.update('none');
    } catch (e) {}
}

async function loadABSummary() {
    if (!activeExperimentId) return;
    try {
        const res = await fetch(`${API_BASE}/ab/experiments/${activeExperimentId}/summary`);
        const data = await res.json();

        document.getElementById('ab-ctr-a').textContent = (data.group_a_ctr * 100).toFixed(2) + '%';
        document.getElementById('ab-impressions-a').textContent = `${data.group_a_impressions} impressions, ${data.group_a_clicks} clicks`;
        document.getElementById('ab-ctr-b').textContent = (data.group_b_ctr * 100).toFixed(2) + '%';
        document.getElementById('ab-impressions-b').textContent = `${data.group_b_impressions} impressions, ${data.group_b_clicks} clicks`;
        document.getElementById('ab-lift').textContent = `+${data.lift_percent.toFixed(1)}%`;
        document.getElementById('ab-pvalue').textContent = data.p_value !== null
            ? `p=${data.p_value.toFixed(4)} ${data.is_significant ? '✓ 显著' : '× 不显著'}`
            : '样本不足';
    } catch (e) {}
}


// =============================================
// Online Fine-tuning
// =============================================

let finetunePollingInterval = null;

async function triggerFinetune() {
    try {
        await fetch(`${API_BASE}/finetune/trigger`, { method: 'POST' });
        pollFinetuneStatus();
    } catch (e) {}
}

async function pollFinetuneStatus() {
    try {
        const res = await fetch(`${API_BASE}/finetune/status`);
        const data = await res.json();
        const bar = document.getElementById('finetune-buffer-bar');
        const status = document.getElementById('finetune-buffer-status');
        const pct = Math.min(100, (data.buffer_size / 500) * 100);
        bar.style.width = pct + '%';
        status.textContent = `Buffer: ${data.buffer_size} / 500${data.is_running ? ' | 微调中...' : ''}`;
    } catch (e) {}
}

async function updateFinetuneConfig() {
    const interval = parseInt(document.getElementById('ft-interval').value);
    const minBuffer = parseInt(document.getElementById('ft-min-buffer').value);
    try {
        await fetch(`${API_BASE}/finetune/config`, {
            method: 'PUT',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ interval_seconds: interval, min_buffer_size: minBuffer })
        });
    } catch (e) {}
}

async function loadFinetuneRuns() {
    try {
        const res = await fetch(`${API_BASE}/finetune/runs`);
        const runs = await res.json();
        let html = '';
        runs.slice(0, 10).forEach(r => {
            const improvement = r.reward_after && r.reward_before
                ? `(${(r.reward_after - r.reward_before).toFixed(3)})`
                : '';
            html += `<div>[${r.status}] #${r.id}: ${r.n_interactions_used}条数据, 奖励变化 ${improvement}</div>`;
        });
        document.getElementById('finetune-runs-list').innerHTML = html || '无记录';
    } catch (e) {}
}


// =============================================
// Performance Benchmark
// =============================================

let perfTimeChart = null;
let perfRewardChart = null;

function initPerfCharts() {
    const defaultOpts = {
        responsive: true,
        plugins: { legend: { labels: { color: '#aaa' } } },
        scales: {
            x: { title: { display: true, text: '数据集大小', color: '#888' }, ticks: { color: '#666' }, grid: { color: '#333' } },
            y: { ticks: { color: '#666' }, grid: { color: '#333' } },
        }
    };

    const ctx1 = document.getElementById('chart-perf-time').getContext('2d');
    perfTimeChart = new Chart(ctx1, {
        type: 'bar',
        data: { labels: [], datasets: [{ label: '训练时间 (秒)', data: [], backgroundColor: '#ff6b6b88', borderColor: '#ff6b6b', borderWidth: 1 }] },
        options: { ...defaultOpts, scales: { ...defaultOpts.scales, y: { ...defaultOpts.scales.y, title: { display: true, text: '时间(s)', color: '#888' } } } }
    });

    const ctx2 = document.getElementById('chart-perf-reward').getContext('2d');
    perfRewardChart = new Chart(ctx2, {
        type: 'line',
        data: { labels: [], datasets: [{ label: '最终奖励', data: [], borderColor: '#6bcf7f', fill: false, tension: 0.3 }] },
        options: { ...defaultOpts, scales: { ...defaultOpts.scales, y: { ...defaultOpts.scales.y, title: { display: true, text: '奖励', color: '#888' } } } }
    });
}

async function startBenchmark() {
    const sizes = document.getElementById('bench-sizes').value.split(',').map(s => parseInt(s.trim()));
    const epochs = parseInt(document.getElementById('bench-epochs').value);
    document.getElementById('bench-status').textContent = '运行中...';

    try {
        await fetch(`${API_BASE}/performance/benchmark/start`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ dataset_sizes: sizes, epochs })
        });
        const pollId = setInterval(async () => {
            const sRes = await fetch(`${API_BASE}/performance/benchmark/status`);
            const status = await sRes.json();
            if (!status.is_running) {
                clearInterval(pollId);
                document.getElementById('bench-status').textContent = '完成';
                loadPerfResults();
            } else {
                document.getElementById('bench-status').textContent = `运行中: ${status.current_size} ...`;
            }
        }, 5000);
    } catch (e) {
        document.getElementById('bench-status').textContent = '错误';
    }
}

async function loadPerfResults() {
    try {
        const res = await fetch(`${API_BASE}/performance/benchmark/results`);
        const data = await res.json();
        const entries = data.entries;

        if (!entries.length) return;

        const labels = entries.map(e => e.dataset_size.toLocaleString());
        const times = entries.map(e => e.training_time_seconds);
        const rewards = entries.map(e => e.final_reward);

        perfTimeChart.data.labels = labels;
        perfTimeChart.data.datasets[0].data = times;
        perfTimeChart.update('none');

        perfRewardChart.data.labels = labels;
        perfRewardChart.data.datasets[0].data = rewards;
        perfRewardChart.update('none');

        let html = '';
        entries.forEach(e => {
            const timePer1K = (e.training_time_seconds / (e.dataset_size / 1000)).toFixed(2);
            html += `<tr>
                <td>${e.dataset_size.toLocaleString()}</td>
                <td>${e.algorithm}</td>
                <td>${e.training_time_seconds.toFixed(1)}</td>
                <td>${e.convergence_epoch || '-'}</td>
                <td>${e.final_reward.toFixed(3)}</td>
                <td>${timePer1K}s</td>
            </tr>`;
        });
        document.getElementById('perf-tbody').innerHTML = html;
    } catch (e) {}
}


// =============================================
// Initialization
// =============================================

document.addEventListener('DOMContentLoaded', () => {
    initCTRChart();
    initPerfCharts();
    loadPolicyInfo();
    loadABExperiments();
    pollFinetuneStatus();
    loadFinetuneRuns();
    loadPerfResults();

    setInterval(pollFinetuneStatus, 10000);
    setInterval(loadFinetuneRuns, 30000);
});
