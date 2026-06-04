const API_BASE = '/api';
const COLORS = {
    cql: '#ff6b6b',
    dqn: '#ffd93d',
    behavior_cloning: '#6bcf7f',
};
const LABELS = {
    cql: 'CQL',
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
    } catch (e) {
        console.error(e);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    refreshRuns();
});
