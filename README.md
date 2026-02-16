# 0DTE Gamma RWR (Radar Warning Receiver)

A high-frequency risk monitoring system for SPX 0DTE options credit spreads. The system detects **Gamma Acceleration Ramps** (G.A.R.) and provides real-time threat levels and confidence-weighted alerts via a terminal HUD.

---

## 🛠 Prerequisites

- **Python 3.12+** (Managed via [uv](https://github.com/astral-sh/uv))
- **Docker** (For running Redis)
- **Massive.com (Polygon.io) API Key** (For real-time options data)
- **Charles Schwab API Credentials** (For position syncing)

---

## 🚀 Local Setup

### 1. Environment Configuration
Create a `.env` file in the root directory:
```env
# Infrastructure
REDIS_HOST=localhost
REDIS_PORT=6379

# Data Provider (Massive.com / Polygon.io)
MASSIVE_API_KEY=your_massive_api_key

# Broker (Charles Schwab)
SCHWAB_API_KEY=your_schwab_key
SCHWAB_API_SECRET=your_schwab_secret
SCHWAB_ACCOUNT_ID=your_account_id
```

### 2. Start Redis
The system requires Redis as a data buffer and message bus.
```bash
# Build the specialized trading-redis image
docker build -t trading-redis -f docker/redis/Dockerfile .

# Run the container
docker run -d --name trading-redis -p 6379:6379 trading-redis
```

### 3. Install Dependencies
```bash
uv sync
```

---

## 📈 Running the System (Live Mode)

To run the live monitoring stack, use two terminal windows. The system uses a unified `BarEvent` pipeline where the producer handles data collection and the consumer handles risk logic.

### Terminal A: Tick Producer (The Data Siphon)
The `TickProducer` automatically monitors all legs defined in `backtest_positions.json` (proxy for live spreads) and publishes unified events.
```bash
.venv\Scripts\python tick_producer.py
```

### Terminal B: Strategy Consumer (The Radar)
The `StrategyConsumer` listens for unified events and renders the RWR HUD.
```bash
.venv\Scripts\python strategy_consumer.py --ticker SPY
```

---

## ⚡ Unified Event Architecture

The system is built on a **Unified BarEvent** schema. This allows the exact same logic (Greeks, GAR, ECCM) to be used for both live trading and historical back-testing.

- **Live Flow**: `Massive (Polygon)` → `TickProducer` → `Redis (tick_events:SPY)` → `StrategyConsumer`
- **Backtest Flow**: `Massive Cache/API` → `ReplayProducer` → `Redis (tick_events:SPY)` → `StrategyConsumer`

---

## ⚡ Dynamic Monitoring

The system supports **Dynamic Symbol Subscriptions**. When the `StrategyConsumer` detects a new credit spread in your account, it automatically signals the `TickProducer` via Redis Pub/Sub (`tick_producer:commands`).

---

## 🧪 Testing & Verification (Backtesting)

### 1. Historical Event Replay (Combined Flow)
This allows you to test the entire pipeline (Greeks, GAR, ECCM, HUD) against historical data streamed through Redis as if it were live.

1. **Define Positions**: Edit `backtest_positions.json` to define the spreads you want to test.
2. **Start Strategy Consumer**:
   ```bash
   .venv\Scripts\python strategy_consumer.py --ticker SPY
   ```
3. **Run Replay Producer**:
   ```bash
   .venv\Scripts\python replay_producer.py --ticker SPY --days 1 --cache
   ```
   *The `ReplayProducer` will stream historical bars into Redis, which the consumer will process in real-time.*

### 2. E2E Logic Test
Run the internal verification suite for HUD rendering and persistence:
```bash
.venv\Scripts\python tests/test_rwr_e2e.py
```

---

## 🏭 Production Deployment

For production environments, ensure the following is configured:

1.  **Process Management**: Use `pm2` or `systemd` to ensure `tick_producer.py` and `strategy_consumer.py` automatically restart on failure.
2.  **Containerization**: Deploy the Redis instance using a managed service or the provided Docker image with a persistent volume for `--save` snapshots.
3.  **Low Latency**: Ensure your production server has low-latency connectivity to Massive.com (Polygon.io) endpoints.
4.  **Logging**: All system logs are written to `strategy_consumer.log` and `tick_producer.log`. Monitor these files for connection issues or API rate limits.

---

## 🛡 System Architecture

- **Data Layer**: Massive/Polygon (WebSocket/Snapshots) → `TickProducer`
- **Internal Bus**: `TickProducer` → Redis (Pub/Sub & ZSet)
- **Risk Engine**: `StrategyConsumer` → `GammaRWREngine` (GAR Calcs)
- **ECCM Logic**: `GammaRWRFilters` (Volume/IV/Persistence Validation)
- **HUD/Alerts**: `RWRAlertManager` (Terminal Rendering & Throttling)
