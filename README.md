# Automated Trading System Backend

A sophisticated, multi-strategy algorithmic trading platform designed for institutional-grade automated execution across futures, equities, and options markets. Built with Python and Flask, this system provides real-time market analysis, multi-broker integration, and comprehensive risk management capabilities.

## Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Trading Strategies](#trading-strategies)
- [Supported Instruments](#supported-instruments)
- [API Integrations](#api-integrations)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [API Documentation](#api-documentation)
- [Security](#security)
- [Deployment](#deployment)
- [Monitoring & Logging](#monitoring--logging)
- [Development Roadmap](#development-roadmap)

## Overview

This automated trading system serves as a comprehensive backend infrastructure for executing sophisticated trading strategies across multiple asset classes. The platform integrates with leading financial institutions and market data providers to deliver high-performance algorithmic trading capabilities.

### Key Capabilities
- **Multi-Strategy Execution**: Simultaneous deployment of EMA crossover, Supertrend reversal, and 0DTE options strategies
- **Multi-Broker Support**: Seamless integration with Charles Schwab and Tastytrade APIs
- **Real-Time Market Data**: High-frequency tick data processing via Databento
- **Risk Management**: Built-in position sizing, stop-loss mechanisms, and trade monitoring
- **RESTful API**: Comprehensive API for strategy management and trade execution
- **Web-Based Interface**: Secure authentication and real-time monitoring capabilities

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │  Mobile Client  │    │  Admin Panel    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                   ┌─────────────┴───────────┐
                   │     Flask API Server    │
                   │   (Authentication &     │
                   │    Strategy Management) │
                   └─────────────┬───────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼───────┐    ┌─────────▼───────┐    ┌─────────▼───────┐
│  EMA Strategy   │    │ Supertrend      │    │  0DTE Options   │
│   Engine        │    │ Strategy        │    │   Strategy      │
│                 │    │ Engine          │    │   Engine        │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                   ┌─────────────▼───────────┐
                   │    Market Data Layer    │
                   │  (Databento + Redis)    │
                   └─────────────┬───────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
┌─────────▼───────┐    ┌─────────▼───────┐    ┌─────────▼───────┐
│ Charles Schwab  │    │   Tastytrade    │    │   Redis Cache   │
│      API        │    │      API        │    │   (Sessions &   │
│                 │    │                 │    │   State Mgmt)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Trading Strategies

### 1. EMA Crossover Strategy
A momentum-based strategy that generates signals when exponential moving averages cross over each other.

**Features:**
- Configurable EMA periods for trend identification
- Support for SMA, EMA, and Wilder's moving averages
- Multi-timeframe analysis (1m to 1d)
- Automatic position sizing and risk management
- Real-time signal generation and execution

**Signal Logic:**
- **Long Entry**: Fast EMA crosses above slow EMA
- **Short Entry**: Fast EMA crosses below slow EMA
- **Exit**: Reverse crossover or stop-loss triggered

### 2. Supertrend Reversal Strategy
Advanced trend-following strategy based on ThinkScript logic with enhanced reversal detection.

**Features:**
- ATR-based volatility adjustment
- Reversal candle signal detection
- Dynamic stop-loss and take-profit levels
- High-frequency execution capabilities
- Customizable risk parameters

**Signal Logic:**
- **Buy Signal**: Price breaks above Supertrend line with bullish reversal candle
- **Sell Signal**: Price breaks below Supertrend line with bearish reversal candle
- **Stop Loss**: ATR-based trailing stop implementation

### 3. 0DTE SPX Options Strategy
Specialized strategy for same-day expiration SPX options trading.

**Features:**
- ATM (At-The-Money) option selection
- EMA crossover signal generation on SPX index
- Automatic option chain analysis
- Manual override capabilities
- Intraday expiration management

**Signal Logic:**
- **Call Options**: SPX EMA bullish crossover
- **Put Options**: SPX EMA bearish crossover
- **Expiration**: Automatic position closure before market close

## Supported Instruments

### Futures Markets
- **Micro E-mini S&P 500** (`/MES`) - Primary equity index futures
- **E-mini S&P 500** (`/ES`) - Standard equity index futures
- **Micro E-mini Nasdaq-100** (`/MNQ`) - Technology-focused index
- **E-mini Nasdaq-100** (`/NQ`) - Standard technology index
- **Micro Russell 2000** (`/M2K`) - Small-cap exposure
- **Russell 2000** (`/RTY`) - Standard small-cap index

### Equity Markets
- **MicroStrategy** (`MSTR`) - Bitcoin proxy equity
- **Tesla** (`TSLA`) - Electric vehicle and energy sector
- **NVIDIA** (`NVDA`) - Semiconductor and AI technology

### Options Markets
- **SPX 0DTE Options** - Same-day expiration S&P 500 index options

## API Integrations

### Broker Integrations
- **Charles Schwab API**
  - OAuth 2.0 authentication
  - Real-time order execution
  - Account management and portfolio tracking
  - Market data access

- **Tastytrade API**
  - Options trading capabilities
  - Advanced order types
  - Real-time position monitoring
  - Risk management tools

### Market Data Providers
- **Databento**
  - High-frequency tick data
  - Historical market data
  - Real-time market feeds
  - Custom data delivery

- **Redis Cache**
  - Session management
  - Strategy state persistence
  - Real-time data buffering
  - Performance optimization

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Redis server
- Valid API credentials for integrated brokers and data providers

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/0x0Zeus/tasty-schwab-trader-BE.git
   cd tasty-schwab-trader-BE
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment configuration:**
   Create a `.env` file in the project root:
   ```bash
   # Broker API Credentials
   SCHWAB_API_KEY=your_schwab_api_key
   SCHWAB_ACCOUNT_ID=your_schwab_account_id
   TASTYTRADE_API_KEY=your_tastytrade_api_key
   
   # Market Data
   DATABENTO_API_KEY=your_databento_api_key
   
   # Redis Configuration
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_DB=0
   
   # Application Settings
   JWT_SECRET=your_secure_jwt_secret
   FLASK_ENV=production
   ```

5. **Initialize Redis:**
   ```bash
   # Start Redis server
   redis-server
   
   # Verify connection
   redis-cli ping
   ```

6. **Run the application:**
   ```bash
   python app.py
   ```

The API server will be available at `http://localhost:5000`

## Configuration

### Strategy Parameters

#### EMA Strategy Configuration
```json
{
  "symbol": "MES",
  "timeframe": "5m",
  "trend_line_1": "EMA",
  "period_1": 9,
  "trend_line_2": "EMA", 
  "period_2": 21,
  "schwab_quantity": 1,
  "tastytrade_quantity": 1,
  "trade_enabled": true
}
```

#### Supertrend Strategy Configuration
```json
{
  "symbol": "ES",
  "timeframe": "1m",
  "schwab_quantity": 1,
  "tastytrade_quantity": 1,
  "trade_enabled": true
}
```

### Timeframe Support
- **Standard Timeframes**: 1m, 2m, 5m, 15m, 30m, 1h, 4h, 1d
- **Custom Tick Charts**: 512t, 1160t, 1600t, etc.
- **Check Intervals**:
  - Short timeframes (1m-5m): 10-second intervals
  - Medium timeframes (15m-1h): 1-minute intervals
  - Tick charts: 5-15 second intervals

## API Documentation

### Authentication
All API endpoints require JWT authentication via Bearer token in the Authorization header:
```
Authorization: Bearer <your_jwt_token>
```

### Core Endpoints

#### Authentication
- `POST /api/login` - User authentication
- `POST /api/update-credentials` - Update admin credentials

#### Broker Integration
- `GET /api/schwab/authorize-url` - Get Schwab OAuth URL
- `POST /api/schwab/access-token` - Exchange authorization code for token
- `POST /api/schwab/refresh-token` - Refresh Schwab access token
- `GET /api/tasty/authorize-url` - Get Tastytrade OAuth URL
- `POST /api/tasty/access-token` - Exchange authorization code for token
- `POST /api/tasty/refresh-token` - Refresh Tastytrade access token

#### Strategy Management
- `POST /api/add-ticker` - Add instrument to strategy
- `GET /api/get-ticker` - Retrieve strategy configuration
- `DELETE /api/delete-ticker` - Remove instrument from strategy
- `GET /api/start-trading` - Start strategy execution
- `GET /api/stop-trading` - Stop strategy execution

#### Manual Trading
- `POST /api/manual-trigger` - Manual 0DTE options trigger
- `POST /api/ema-manual-trigger` - Manual EMA strategy trigger

### Example API Usage

#### Starting EMA Strategy
```bash
curl -X GET "http://localhost:5000/api/start-trading?strategy=ema" \
  -H "Authorization: Bearer <your_token>"
```

#### Adding Instrument to Strategy
```bash
curl -X POST "http://localhost:5000/api/add-ticker" \
  -H "Authorization: Bearer <your_token>" \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "ema",
    "symbol": "MES",
    "timeframe": "5m",
    "trend_line_1": "EMA",
    "period_1": 9,
    "trend_line_2": "EMA",
    "period_2": 21,
    "schwab_quantity": 1,
    "tastytrade_quantity": 1,
    "trade_enabled": true
  }'
```

## Security

### Authentication & Authorization
- JWT-based authentication with configurable expiration
- Secure credential storage and management
- Role-based access control for administrative functions

### Data Protection
- Environment variable configuration for sensitive data
- Secure token storage and refresh mechanisms
- Encrypted communication with broker APIs

### Best Practices
- Regular credential rotation
- Secure API key management
- Comprehensive logging for audit trails
- Input validation and sanitization

## Deployment

### Production Environment
- **Platform**: AWS EC2 instances
- **Process Management**: systemd supervisor for auto-restart
- **Load Balancing**: Application Load Balancer for high availability
- **Monitoring**: CloudWatch integration for performance metrics

### Infrastructure Requirements
- **Minimum**: 2 vCPU, 4GB RAM
- **Recommended**: 4 vCPU, 8GB RAM
- **Storage**: 50GB SSD for logs and data
- **Network**: Stable internet connection for real-time data

### Deployment Steps
1. Launch EC2 instance with appropriate specifications
2. Install Python 3.8+, Redis, and systemd
3. Clone repository and configure environment variables
4. Set up systemd service for automatic startup
5. Configure CloudWatch logging and monitoring
6. Set up SSL certificates for secure communication

## Monitoring & Logging

### Log Management
- **Strategy Logs**: Individual log files per instrument and strategy
- **Application Logs**: Centralized application and error logging
- **Trade Logs**: Comprehensive trade execution and performance tracking

### Log Locations
```
logs/
├── ema/           # EMA strategy logs by instrument
├── supertrend/    # Supertrend strategy logs
├── zeroday/       # 0DTE options strategy logs
└── strategy_consumer.log  # Main application log
```

### Performance Monitoring
- Real-time strategy performance tracking
- Trade execution latency monitoring
- System resource utilization alerts
- Broker API response time tracking

## Development Roadmap

### Phase 1: Performance Optimization
- [ ] Reduce execution latency through async processing
- [ ] Implement connection pooling for broker APIs
- [ ] Optimize data processing pipelines
- [ ] Add hardware acceleration support

### Phase 2: Enhanced Risk Management
- [ ] Implement daily loss limits per strategy
- [ ] Add position sizing algorithms
- [ ] Develop correlation-based risk controls
- [ ] Create real-time risk monitoring dashboard

### Phase 3: Strategy Expansion
- [ ] Add mean reversion strategies
- [ ] Implement machine learning-based signals
- [ ] Develop options volatility strategies
- [ ] Create portfolio optimization algorithms

### Phase 4: Platform Enhancement
- [ ] Multi-broker execution capabilities
- [ ] Advanced backtesting framework
- [ ] Paper trading simulation environment
- [ ] Mobile application development

### Phase 5: Institutional Features
- [ ] Compliance and regulatory reporting
- [ ] Advanced order management system
- [ ] Portfolio analytics and reporting
- [ ] Integration with institutional data providers

---

## License

This project is proprietary software. All rights reserved.

## Support

For technical support and inquiries, please contact the development team through the appropriate channels.

---

**Disclaimer**: This software is for educational and research purposes. Trading involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results.