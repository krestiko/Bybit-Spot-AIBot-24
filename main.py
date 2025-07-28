import time
import os
import logging
from collections import deque
import numpy as np
from sklearn.linear_model import SGDClassifier
import joblib
from pybit.unified_trading import HTTP
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")
symbol = os.getenv("SYMBOL", "BTCUSDT")
trade_amount = float(os.getenv("TRADE_AMOUNT_USDT", 10))
tp_percent = float(os.getenv("TP_PERCENT", 1.5))
sl_percent = float(os.getenv("SL_PERCENT", 1.0))
interval = int(os.getenv("INTERVAL", 5)) * 60
max_retries = int(os.getenv("MAX_RETRIES", 3))

# length of price history to use for learning
history_len = int(os.getenv("HISTORY_LENGTH", 10))

telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

# file to store the trained model
model_file = os.getenv("MODEL_FILE", "model.pkl")

# containers for price history and ML model
price_history = deque(maxlen=history_len + 1)
if os.path.exists(model_file):
    model = joblib.load(model_file)
    model_initialized = True
else:
    model = SGDClassifier(loss="log_loss")
    model_initialized = False

# price at which current position was opened; None means no position
position_price = None

def send_telegram(msg):
    if not telegram_token or not telegram_chat_id: return
    import requests
    try:
        requests.post(f"https://api.telegram.org/bot{telegram_token}/sendMessage",
                      data={"chat_id": telegram_chat_id, "text": msg})
    except Exception as e:
        logging.warning(f"Telegram error: {e}")

session = HTTP(api_key=api_key, api_secret=api_secret)

def get_price():
    for _ in range(max_retries):
        try:
            data = session.get_tickers(category="spot", symbol=symbol)
            return float(data["result"]["list"][0]["lastPrice"])
        except Exception as e:
            logging.warning(f"Price fetch error: {e}")
            time.sleep(1)
    raise RuntimeError("Failed to fetch price")

def extract_features(history):
    arr = np.array(history, dtype=float)
    returns = np.diff(arr) / arr[:-1]
    sma = np.mean(arr)
    volatility = np.std(arr)
    extra = np.array([arr[-1] - sma, volatility])
    return np.concatenate([returns, extra]).reshape(1, -1)

def update_model(new_price):
    global model_initialized
    price_history.append(new_price)
    if len(price_history) < history_len + 1:
        return None
    features = extract_features(list(price_history)[:-1])
    label = 1 if price_history[-1] > price_history[-2] else 0
    model.partial_fit(features, [label], classes=[0, 1])
    joblib.dump(model, model_file)
    model_initialized = True
    return extract_features(list(price_history)[-history_len:])
 
def check_balance(side):
    try:
        bal = session.get_wallet_balance(accountType="UNIFIED")
        coins = {c["coin"]: float(c.get("availableToTrade", 0))
                 for c in bal["result"]["list"][0]["coin"]}
        base = symbol.replace("USDT", "")
        if side == "Buy":
            return coins.get("USDT", 0) >= trade_amount
        else:
            return coins.get(base, 0) > 0
    except Exception as e:
        logging.warning(f"Balance check error: {e}")
        return True

def place_order(side, price):
    for _ in range(max_retries):
        try:
            resp = session.place_order(
                category="spot",
                symbol=symbol,
                side=side,
                order_type="Market",
                quote_qty=trade_amount,
                time_in_force="IOC",
            )
            if resp.get("retCode") != 0:
                raise Exception(resp.get("retMsg"))
            order_id = resp["result"]["orderId"]
            for _ in range(5):
                info = session.get_order_history(category="spot", orderId=order_id)
                status = info["result"]["list"][0]["orderStatus"]
                if status == "Filled":
                    msg = f"{side} executed {symbol} at {price}"
                    logging.info(msg)
                    send_telegram(msg)
                    return True
                time.sleep(1)
        except Exception as e:
            logging.warning(f"Order error: {e}")
            time.sleep(1)
    logging.error("Failed to place order")
    send_telegram("Failed to place order")
    return False

def trade_loop():
    logging.info(f"Bot start: {symbol}, {trade_amount} USDT per trade")
    send_telegram(f"Bot live: trading {symbol}, every {interval//60} min")
    global position_price
    while True:
        try:
            price = get_price()
            logging.info(f"Price: {price}")
            features = update_model(price)
            if position_price is not None:
                if price >= position_price * (1 + tp_percent / 100):
                    if place_order("Sell", price):
                        send_telegram(f"Take profit at {price}")
                        position_price = None
                    continue
                if price <= position_price * (1 - sl_percent / 100):
                    if place_order("Sell", price):
                        send_telegram(f"Stop loss at {price}")
                        position_price = None
                    continue

            if model_initialized and features is not None:
                prediction = model.predict(features)[0]
                decision = "Buy" if prediction == 1 else "Sell"
            else:
                if len(price_history) >= 2:
                    avg = sum(price_history) / len(price_history)
                    decision = "Buy" if price > avg else "Sell"
                else:
                    decision = "Buy" if int(price) % 2 == 0 else "Sell"

            if position_price is None and decision == "Buy":
                if check_balance("Buy") and place_order("Buy", price):
                    position_price = price
            elif position_price is not None and decision == "Sell":
                if check_balance("Sell") and place_order("Sell", price):
                    position_price = None
        except Exception as e:
            logging.error(f"Loop error: {e}")
            send_telegram(f"Bot error: {e}")
        time.sleep(interval)

if __name__ == "__main__":
    trade_loop()
