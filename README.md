# Offline RL Recommendation Strategy Simulator

基于离线强化学习的电商推荐系统策略模拟器。实现CQL (Conservative Q-Learning)算法，对比传统DQN和Behavior Cloning的效果。

## Architecture

- **Backend:** FastAPI + PyTorch + SQLAlchemy + PostgreSQL
- **Frontend:** HTML + Chart.js (实时学习曲线可视化)
- **Infrastructure:** Docker Compose

## Quick Start

```bash
# 启动所有服务
docker-compose up -d

# 等待数据库就绪后，访问前端
# http://localhost:3000

# 或通过API操作:
# 1. 生成离线数据
curl -X POST http://localhost:8000/api/data/generate

# 2. 启动CQL训练
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "cql", "hyperparameters": {"alpha": 1.0, "epochs": 200}}'

# 3. 启动DQN训练 (对比基线)
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "dqn", "hyperparameters": {"epochs": 200}}'

# 4. 启动Behavior Cloning (对比基线)
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"algorithm": "behavior_cloning", "hyperparameters": {"epochs": 200}}'
```

## Environment Design

- **State:** 10维用户兴趣向量 (10个商品类别)
- **Action:** 从100个物品中推荐一个 (每类10个)
- **Reward:** 点击=1.0, 未点击=0.0
- **Episode:** 每用户会话50步

## CQL Algorithm

CQL在标准DQN的Bellman损失基础上添加保守惩罚项：

```
L_total = L_bellman + α × (E_s[logsumexp(Q(s,·))] - E_{(s,a)~D}[Q(s,a)])
```

避免对离线数据中未出现的动作产生过高的Q值估计。

## Key Files

| File | Description |
|------|-------------|
| `backend/algorithms/cql.py` | CQL核心算法 |
| `backend/algorithms/dqn.py` | DQN基线 |
| `backend/algorithms/behavior_cloning.py` | 行为克隆基线 |
| `backend/environment/simulator.py` | 推荐环境模拟器 |
| `backend/data/generator.py` | 离线数据生成 |
| `backend/api/` | REST API + WebSocket |
| `frontend/` | 可视化仪表板 |
