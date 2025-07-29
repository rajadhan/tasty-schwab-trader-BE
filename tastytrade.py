import requests
from time import sleep
from config import *
import pandas as pd
from datetime import datetime
import pytz
import os

def generate_access_token_for_tastytrade():
    access_token_url = f'{tastytrade_link}/sessions'
    payload = {
        "login": username,
        "password": password,
        "remember-me": True
    }
    try:
        response = requests.post(url = access_token_url, json = payload)
        access_token = response.json()['data']['session-token']
        with open(tastytrade_access_token_path, 'w') as f:
            f.write(access_token)
    except Exception as e:
        print(f'ERROR! in generating access token for {username} = {e}')
        sleep(5)
        generate_access_token_for_tastytrade(username, password)

def create_header(token):
    return {"Accept": "application/json", "Content-Type": "application/json", "Authorization": token}

def create_order_payload(symbol, qty, action, account_id, logger):
    instrument_type = 'Future' if symbol[0] == '/' else 'Equity'
    order_payload = {
                    "order-type": 'Market',
                    "time-in-force": 'Day',
                    "legs": [
                        {
                        "instrument-type": instrument_type,
                        "symbol": symbol,
                        "action": action,
                        "quantity": str(qty)
                        }
                    ]
                    }

    logger.info(f'{account_id}=> Credit order payload: {order_payload}')
    return order_payload
        
     
def place_order(order_payload, account_id, access_token, logger):
    place_order_url = f'{tastytrade_link}/accounts/{account_id}/orders'
    try:
        response = requests.post(url = place_order_url,
            json = order_payload,
            headers = create_header(access_token)
        )
        order = response.json()
        logger.info(f'{account_id}=> order response = {order}')
        order_id = order['data']['order']['id']
        return order_id
    except Exception as e:
        logger.error(f'{account_id}=> ERROR! in placing order = {e}. Response text = {response.text}')
        return ""

        
def check_order_status(order_id, account_id, access_token, logger):
    try:
        check_order_status_url = f'{tastytrade_link}/accounts/{account_id}/orders/{order_id}'
        response = requests.get(url = check_order_status_url, headers = create_header(access_token))
        order_history = response.json()

        logger.info(f'order_history for user {account_id} = {order_history}')
        order_status = order_history['data']['status']
        traded_qty = int(order_history['data']['size'])
        if order_status == 'Rejected' or order_status == 'Cancelled':
            return False, traded_qty
        else:
            if order_status == 'Filled':
                return True, traded_qty
            else:
                sleep(1)
                is_filled, traded_qty = check_order_status(order_id, account_id, access_token, logger)
                return is_filled, traded_qty

    except Exception as e:
        logger.error(f'{account_id}=> ERROR! in checking order status = {str(e)}. Response text = {response.text}')
        return False, 0

def place_tastytrade_order(symbol, qty, action, account_id, logger):
    try:
        with open(tastytrade_access_token_path, "r") as file:
            access_token = file.read()
        if symbol[0] == '/':
            instrument_df = pd.read_csv("tastytrade_instruments.csv")
            df = instrument_df[(instrument_df["product-code"] == symbol[1:]) & (instrument_df["active-month"] == True)][["exchange-symbol", "expires-at"]]
            expiry = pd.to_datetime(df["expires-at"].values[0])
            
            # Convert expiry to US/Eastern timezone
            eastern = pytz.timezone('US/Eastern')
            expiry_eastern = expiry.astimezone(eastern)

            # Convert datetime.now() to US/Eastern timezone
            now_eastern = datetime.now(pytz.utc).astimezone(eastern)
            if expiry_eastern <= now_eastern:
                symbol = instrument_df[(instrument_df["product-code"] == symbol[1:]) & (instrument_df["next-active-month"] == True)]["exchange-symbol"].tolist()[0]
                symbol = "/" + symbol
            else:
                symbol = "/" + df["exchange-symbol"].values[0]
        if symbol[-1].isdigit():   #check for futrues contract if true add /
            symbol = f"/{symbol.lstrip('/')}"
        order_payload = create_order_payload(symbol, qty, action, account_id, logger)
        order_id = place_order(order_payload, account_id, access_token, logger)
        is_filled, _ = check_order_status(order_id, account_id, access_token, logger)
        if is_filled:
            return order_id
        else:
            logger.warning(f"Order not filled for {symbol}. Order ID: {order_id}")
            logger.warning(f"Placing order again for {symbol}. Order ID: {order_id}")
            order_id = place_order(order_payload, account_id, access_token, logger)
            return order_id
    except Exception as e:
        print(f'{account_id}=> ERROR! in placing order = {e}.')
        return ""

def get_instruments():
    try:
        with open(tastytrade_access_token_path, "r") as file:
            access_token = file.read()
        hearder = create_header(access_token)
        url = f'{tastytrade_link}/instruments/futures'
        response = requests.get(url = url, headers = hearder)
        instruments = response.json()["data"]["items"]
        df = pd.DataFrame(instruments)
        df.to_csv("tastytrade_instruments.csv")
    except Exception as e:
        print(f'ERROR! in getting instruments = {e}')
    

# class TastyTrade:

#     @staticmethod
#     def getOptionChain(symbol, access_token):
#         try:
#             endpoint = f'{tastytrade_link}/option-chains/{symbol}'

#             response = requests.get(url = endpoint, headers=TastyTrade.create_header(access_token))
#             if response.status_code != 200:
#                 print(f"ERROR! in getting option chain from tastytrade for {symbol} = {response.content}")
#                 return {}

#             option_chain = response.json()
#             return option_chain
#         except Exception as e:
#             print(f'ERROR! in getting option chain from tastytrade for {symbol} = {e}')
#             return {}


#     @staticmethod
#     def getOptionSymbol(option_chain, chain, strikePrice, expiry):
#         for option_chain_details in option_chain['data']['items']:
#             if option_chain_details['strike-price'] == str(float(strikePrice)) and option_chain_details['option-type'] == chain and option_chain_details['expiration-date'] == expiry and option_chain_details['root-symbol'] == 'SPXW':
#                 option_symbol = option_chain_details['symbol']
#                 return option_symbol


#     @staticmethod
#     def cancel_order(order_id, account_id, access_token):
#         try:
#             cancel_order_url = f'{tastytrade_link}/accounts/{account_id}/orders/{order_id}'
#             response = requests.delete(url = cancel_order_url, headers = TastyTrade.create_header(access_token))
#             print(f'{account_id}=> Cancel order response = {response.text}')

#         except Exception as e:
#             print(f'{account_id}=> ERROR! in cancelling order = {str(e)}. Response text = {response.text}')