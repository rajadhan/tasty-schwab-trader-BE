# Automated Trading System (Backend)

This project is a Python-based trading engine designed for automated execution of multiple trading strategies across **futures, stocks, and 0DTE SPX options**.  
It integrates with **Charles Schwab** and **Tastytrade APIs** for execution and uses **Databento** for real-time market data.

---

## 🚀 Features
- **EMA Crossover Strategy**
  - Long/short entry when EMAs cross.
  - Supports SMA, EMA, and Wilder’s averages.
  - Works on multiple instruments and timeframes simultaneously.

- **Supertrend Reversal Strategy**
  - Derived from ThinkScript logic.
  - Executes on reversal candle signals (buy/sell).
  - Includes stop-loss and reversal handling.

- **0DTE SPX Options Strategy**
  - Trades ATM SPX options expiring the same day.
  - Signal: EMA crossovers on SPX index.
  - Supports **automatic execution** and **manual push-button trigger**.

---

## 📊 Supported Instruments
- **Futures:** `/MES`, `/ES`, `/MNQ`, `/NQ`, `/M2K`, `/RTY`
- **Stocks:** `MSTR`, `TSLA`, `NVDA`
- **Options:** SPX 0-day-to-expiry (0DTE)

---

## 🔌 Integrations
- **Brokers/APIs**
  - Charles Schwab API
  - Tastytrade API
- **Market Data**
  - Databento (tick-level futures data)
  - Optional: Tastytrade tick feed

---

## ⚙️ Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/trading-backend.git
   cd trading-backend
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Configure API keys:
   ```bash
   SCHWAB_API_KEY=your_key
   SCHWAB_ACCOUNT_ID=your_account
   TASTYTRADE_API_KEY=your_key
   DATABENTO_API_KEY=your_key
5. Run trading engine:
   ```bash
   flask run

---

## 🛠 Configuration
- **Timeframes**: 1m, 2m, 5m, 15m, 30m, 1h, 4h, 1d
- **Tick Charts**: customizable (e.g., 512t, 1160t, 1600t)
- **Check Intervals:**
   - Short TFs: every 10s
   - Medium TFs: every 1m
   - Tick charts: 5–15s

---

## 📡 Deployment
- Runs on **AWS EC2** instance.
- Auto-restarts with **systemd supervisor**.
- Logs stored in ```logs/``` directory.

---

## 📌 Roadmap
- Improve execution latency.
- Expand to multi-broker support.
- Integrate risk management (max loss per day).
- Enhanced backtesting module.