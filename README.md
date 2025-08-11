# Trading Application Backend

A Flask-based trading application that integrates with Schwab and Tastytrade APIs for automated trading strategies.

## Features

- **RESTful API**: Flask-based API with JWT authentication
- **Multi-Broker Support**: Integration with Schwab and Tastytrade
- **Automated Trading**: Configurable trading strategies with EMA/SMA indicators
- **Real-time Data**: Historical market data retrieval
- **Position Management**: Automated position opening/closing based on strategy signals

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure credentials**:
   - Update `config.py` with your API credentials
   - Update `credentials/admin_credentials.json` with your admin login details

## Quick Start

### Option 1: Using the startup script (Recommended)
```bash
python run_app.py
```

### Option 2: Direct Flask run
```bash
python app.py
```

The application will start on `http://localhost:5000`

## API Endpoints

### Authentication
- `POST /api/login` - Login with admin credentials

### Trading Management
- `POST /api/add-ticker` - Add a new ticker with strategy parameters
- `GET /api/get-ticker` - Get all configured tickers
- `POST /api/manual-trade` - Manually execute trades for the zeroday strategy
- `GET /api/start-trading` - Start the automated trading process

## Configuration

### Ticker Configuration Format
Each ticker is configured with the following parameters:
```json
{
  "symbol": "timeframe",
  "schwab_quantity": "trade_enabled",
  "tastytrade_quantity": "trend_line_1",
  "period_1": "trend_line_2",
  "period_2": ""
}
```

### Strategy Parameters
- **timeframe**: Trading interval (e.g., "1Min", "1Hour", "1Day")
- **schwab_quantity**: Number of shares/contracts for Schwab
- **trade_enabled**: "TRUE" or "FALSE" to enable/disable trading
- **tastytrade_quantity**: Number of shares/contracts for Tastytrade
- **trend_line_1/2**: Technical indicators ("EMA", "SMA", "SuperTrend")
- **period_1/2**: Periods for the technical indicators

## File Structure

```
backend/
├── app.py                 # Main Flask application
├── config.py             # Configuration and credentials
├── main_equities.py      # Trading strategy implementation
├── utils.py              # Utility functions
├── tastytrade.py         # Tastytrade API integration
├── schwab/               # Schwab API integration
├── consts/               # Constants and reference data
├── credentials/          # Admin credentials
├── settings/             # Trading configuration
├── tokens/               # API tokens storage
├── jsons/                # JSON configuration files
├── trades/               # Trade history files
├── logs/                 # Application logs
└── previous_logs/        # Archived logs
```

## Testing

Run the import test to verify all dependencies are correctly installed:
```bash
python test_imports.py
```

## Security Notes

- **JWT Secret**: Change the `JWT_SECRET` in `app.py` for production use
- **Credentials**: Never commit API credentials to version control
- **HTTPS**: Use HTTPS in production environments

## Troubleshooting

1. **Import Errors**: Run `python test_imports.py` to check dependencies
2. **Missing Files**: The startup script will create necessary directories and files
3. **API Errors**: Check your credentials in `config.py`
4. **Permission Errors**: Ensure write permissions for logs and data directories

## Development

For development, the application runs in debug mode with auto-reload disabled to prevent conflicts with the trading threads.

## License

This project is for educational and personal use only. Please ensure compliance with your broker's terms of service and local regulations.
