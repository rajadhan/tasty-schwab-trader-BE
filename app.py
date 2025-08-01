from flask import Flask, request, jsonify
from flask_cors import CORS
import json, os, jwt, datetime
from functools import wraps
from main_equities import run_every_week
import sys
import requests
from schwab.client import *

app = Flask(__name__)
CORS(app)

# ========= Secret Key (keep this secret and safe!) =========
JWT_SECRET = 'secret-trading'  # Use a secure random string in production
JWT_EXPIRATION_MINUTES = 60 * 24 * 30

# ================== File Paths ===================
ADMIN_CREDENTIALS_PATH = os.path.join('credentials', 'admin_credentials.json')
SYMBOL_DATA_PATH = os.path.join('consts', 'symbol.json')
TREND_DATA_PATH = os.path.join('consts', 'trend_line.json')
EMA_TICKER_DATA_PATH = os.path.join('settings', 'ema_ticker_data.json')
ZERODAY_TICKER_DATA_PATH = os.path.join('settings', 'zeroday_ticker_data.json')
SUPER_TICKER_DATA_PATH = os.path.join('settings', 'super_ticker_data.json')
REFRESH_TOKEN_LINK = os.path.join('jsons', 'refresh_token_link.json')

# ================== JSON Utilities ===============
def load_json(filepath):
    try:
        with open(filepath, 'r') as file:
            return json.load(file)
    except Exception:
        return {}

def save_json(filepath, data):
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

# ================== Load Initial Data ===============
admin_credentials = load_json(ADMIN_CREDENTIALS_PATH)
symbol_data = load_json(SYMBOL_DATA_PATH)
trend_data = load_json(TREND_DATA_PATH)

# ================== JWT Auth Utilities ===================

def generate_token(email, password):
    payload = {
        'email': email,
        'password': password,
        'exp': datetime.utcnow() + timedelta(minutes=JWT_EXPIRATION_MINUTES)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    return token

def token_required(f):
    @wraps(f)
    def decorator(*args, **kwargs):
        token = None
        # Expecting Authorization: Bearer <token>
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header.replace('Bearer ', '')
        if not token:
            return jsonify({'success': False, 'error': 'Token is missing'}), 401

        try:
            data = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            return jsonify({'success': False, 'error': 'Token has expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401

        return f(*args, **kwargs)
    return decorator

# ================== Routes ==================

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        email = data.get('email')
        password = data.get('password')

        if email == admin_credentials.get('email') and password == admin_credentials.get('password'):
            token = generate_token(email, password)

            refresh_token_path = load_json(REFRESH_TOKEN_LINK);
            refresh_token_link = refresh_token_path.get('refresh_token_link', '')
            return jsonify({'success': True, 'token': token, 'refreshToken': refresh_token_link}), 200
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401
    except Exception as e:
        print("error", e)
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/add-ticker', methods=['POST'])
@token_required
def add_ticker():
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        if (data.get("strategy") == "ema" or data.get("strategy") == "zeroday"):
            if (data.get('symbol') in symbol_data.get('symbol') and 
                data.get('trend_line_1') in trend_data.get('trend') and 
                data.get('trend_line_2') in trend_data.get('trend') and 
                data.get('period_1') and data.get('period_2') and 
                data.get('timeframe') and 
                # data.get('schwab_quantity') and 
                # data.get('tastytrade_quantity') and 
                isinstance(data.get('trade_enabled'), bool)):

                # Convert timeframe to desired format
                raw_timeframe = str(data.get('timeframe'))
                if raw_timeframe.endswith("Min"):
                    formatted_timeframe = raw_timeframe.replace("Min", "")
                elif raw_timeframe.endswith("Hour"):
                    formatted_timeframe = raw_timeframe.replace("Hour", "h")
                elif raw_timeframe.endswith("Day"):
                    formatted_timeframe = raw_timeframe.replace("Day", "d")
                else:
                    formatted_timeframe = raw_timeframe  # e.g., 100t remains unchanged

                symbol_key = f"{data.get('symbol')}"
                formatted = [
                    formatted_timeframe,
                    str(data.get('schwab_quantity')),
                    str(data.get('trade_enabled')).upper(),
                    str(data.get('tastytrade_quantity')),
                    str(data.get('trend_line_1')),
                    str(data.get('period_1')),
                    str(data.get('trend_line_2')),
                    str(data.get('period_2'))
                ]

            else:
                return jsonify({'success': False, 'error': 'Invalid input data'}), 400
        elif data.get("strategy") == "supertrend":
            # Convert timeframe to desired format
            raw_timeframe = str(data.get('timeframe'))
            if raw_timeframe.endswith("Min"):
                formatted_timeframe = raw_timeframe.replace("Min", "")
            elif raw_timeframe.endswith("Hour"):
                formatted_timeframe = raw_timeframe.replace("Hour", "h")
            elif raw_timeframe.endswith("Day"):
                formatted_timeframe = raw_timeframe.replace("Day", "d")
            else:
                formatted_timeframe = raw_timeframe  # e.g., 100t remains unchanged
            symbol_key = f"{data.get('symbol')}"
            formatted = [
                formatted_timeframe,
                str(data.get('schwab_quantity')),
                str(data.get('trade_enabled')).upper(),
                str(data.get('tastytrade_quantity')),
                str(data.get('short_ma_length')),
                str(data.get('short_ma_type')),
                str(data.get('mid_ma_length')),
                str(data.get('mid_ma_type')),
                str(data.get('long_ma_length')),
                str(data.get('long_ma_type')),
                str(data.get('zigzag_percent_reversal')),
                str(data.get('atr_length')),
                str(data.get('zigzag_atr_multiple')),
                str(data.get('fibonacci_enabled')),
                str(data.get('support_demand_enabled')),
            ]
        # Load current data
        if (data.get("strategy") =="ema"):
            saved_data = load_json(EMA_TICKER_DATA_PATH)
            saved_data[symbol_key] = formatted
            save_json(EMA_TICKER_DATA_PATH, saved_data)
        elif (data.get("strategy") == "zeroday"):
            saved_data = load_json(ZERODAY_TICKER_DATA_PATH)
            saved_data[symbol_key] = formatted
            save_json(ZERODAY_TICKER_DATA_PATH, saved_data)
        elif (data.get("strategy") == "supertrend"):
            saved_data = load_json(SUPER_TICKER_DATA_PATH)
            saved_data[symbol_key] = formatted
            save_json(SUPER_TICKER_DATA_PATH, saved_data)
        return jsonify({'success': True, 'data': saved_data}), 201

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/get-ticker', methods=['GET'])
@token_required
def get_ticker():
    try:
        strategy = request.args.get('strategy') # Get query from request

        if strategy == 'ema': # In case of EMA crossover strategy
            data = load_json(EMA_TICKER_DATA_PATH)
        elif strategy == 'zeroday': # In case of 0 day SPX strategy
            data = load_json(ZERODAY_TICKER_DATA_PATH)
        elif strategy == 'supertrend': # In case of Supertrend
            data = load_json(SUPER_TICKER_DATA_PATH)
            
        # In case of no data
        if not data:
            return jsonify({'success': True, 'data': []}), 200

        return jsonify({'success': True, 'data': data}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/delete-ticker', methods=['DELETE'])
@token_required
def delete_ticker():
    try:
        data = request.get_json()
        strategy = data.get('strategy')
        symbol_key = data.get('symbol')

        if strategy == 'ema':
            saved_data = load_json(EMA_TICKER_DATA_PATH)
        elif strategy == 'zeroday':
            saved_data = load_json(ZERODAY_TICKER_DATA_PATH)
        elif strategy == 'supertrend':
            saved_data = load_json(SUPER_TICKER_DATA_PATH)
            
        # in case of empty data
        if not saved_data:
            return jsonify({'success': True, 'data': []}), 200
        
        # Check if symbol exists in data
        if symbol_key not in saved_data:
            return jsonify({'success': False, 'error': 'Symbol not found'}), 404
        
        # Delete the symbol from data
        del saved_data[symbol_key]
        
        # Save the updated data back to file
        if strategy == 'ema':
            save_json(EMA_TICKER_DATA_PATH, saved_data)
        elif strategy == 'zeroday':
            save_json(ZERODAY_TICKER_DATA_PATH, saved_data)
        elif strategy == 'supertrend':
            save_json(SUPER_TICKER_DATA_PATH, saved_data)
        
        return jsonify({'success': True, 'data': saved_data}), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/refresh-token-link', methods=['POST'])
@token_required
def refresh_link():
    data = request.json
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    link = data.get('refresh_token_link')
    success, message = validate_refresh_link(link, REFRESH_TOKEN_LINK)
    return jsonify({"success": success, "message": message})

@app.route('/api/start-trading', methods=['GET'])
@token_required
def start_trading():
    try:
        data = load_json(EMA_TICKER_DATA_PATH)
        if not data:
            return jsonify({'success': False, 'error': 'No ticker data found'}), 404

        trade_enabled_symbols = []

        for symbol, values in data.items():
            # values[2] is trade_enabled ("TRUE" or "FALSE")
            if len(values) >= 3 and values[2] == "TRUE":
                trade_enabled_symbols.append(symbol)

        if trade_enabled_symbols:
            print('Loading trading parameters ... ')            
            run_every_week()       
            # result = requests.get('https://api.schwabapi.com/v1/oauth/authorize?response_type=code&client_id=1iSr8ykD9qh2M2HoQv56wM2R1kWgQYZI&redirect_uri=https://127.0.0.1', allow_redirects=True) 
            print('Trading started!')
            # print(result.json())
            
            # # Get the final URL after all redirects
            # final_url = result.url
            # print(f'Final redirected URL: {final_url}')
            # auth_code = result.args.get('code')
            # print(f'Auth code: {auth_code}')
            # header = create_header("Bearer")
            return jsonify({
                'success': True,
                'message': 'Trading has started',
                'enabled_symbols': trade_enabled_symbols,
                # 'final_url': final_url,
                # 'status_code': result.status_code
            }), 200
        else:
            return jsonify({
                'success': True,
                'message': 'Trade disabled or no valid tickers with trading enabled'
            }), 200

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
