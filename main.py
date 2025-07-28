import time
import os
import logging
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pandas as pd
import pandas_ta as ta
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
history_len = int(os.getenv("HISTORY_LENGTH", 100))
price_file = os.getenv("PRICE_HISTORY_FILE", "price_history.csv")
trade_file = os.getenv("TRADE_HISTORY_FILE", "trade_history.csv")
trailing_percent = float(os.getenv("TRAILING_PERCENT", 0))
model_type = os.getenv("MODEL_TYPE", "gb")

telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

# file to store the trained model
model_file = os.getenv("MODEL_FILE", "model.pkl")

# containers for market history and ML model
history_df = pd.DataFrame(columns=["close", "volume", "bid_qty", "ask_qty"])
features_list = []
labels_list = []
scaler = StandardScaler()
if os.path.exists(model_file):
    saved = joblib.load(model_file)
    model = saved.get("model")
    scaler = saved.get("scaler", scaler)
    model_initialized = True
else:
    if model_type == "gb":
        model = GradientBoostingClassifier()
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

def get_market_data():
    for _ in range(max_retries):
        try:
            tick = session.get_tickers(category="spot", symbol=symbol)
            ticker = tick["result"]["list"][0]
            price = float(ticker["lastPrice"])
            volume = float(ticker.get("turnover24h", 0))
            ob = session.get_orderbook(category="spot", symbol=symbol, limit=1)
            bid_qty = float(ob["result"]["b"][0][1])
            ask_qty = float(ob["result"]["a"][0][1])
            return price, volume, bid_qty, ask_qty
        except Exception as e:
            logging.warning(f"Market data error: {e}")
            time.sleep(1)
    raise RuntimeError("Failed to fetch market data")

def compute_features(df):
    df = df.copy()
    df["rsi"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"])
    df["macd"] = macd["MACD_12_26_9"]
    bb = ta.bbands(df["close"], length=5)
    df["bb_upper"] = bb["BBU_5_2.0"]
    df["bb_lower"] = bb["BBL_5_2.0"]
    row = df.iloc[-1]
    if row[["rsi", "macd", "bb_upper", "bb_lower"]].isna().any():
        return None
    feats = row[["rsi", "macd", "bb_upper", "bb_lower", "volume", "bid_qty", "ask_qty"]].values
    return feats.reshape(1, -1)

def append_market_data(price, volume, bid, ask):
    global history_df
    history_df.loc[len(history_df)] = [price, volume, bid, ask]
    if len(history_df) > history_len + 50:
        history_df = history_df.iloc[-(history_len + 50):]
    history_df.to_csv(price_file, index=False)

def update_model(price, volume, bid, ask):
    global model_initialized
    append_market_data(price, volume, bid, ask)
    if len(history_df) < history_len + 1:
        return None
    df = history_df.iloc[-(history_len + 1):]
    features = compute_features(df.iloc[:-1])
    if features is None:
        return None
    label = 1 if df["close"].iloc[-1] > df["close"].iloc[-2] else 0
    features_list.append(features.flatten())
    labels_list.append(label)
    X = np.array(features_list)
    y = np.array(labels_list)
    scaler.fit(X)
    Xs = scaler.transform(X)
    try:
        model.fit(Xs, y)
        if len(y) >= 3:
            score = cross_val_score(model, Xs, y, cv=3).mean()
            logging.info(f"CV score: {score:.3f}")
    except Exception as e:
        logging.warning(f"Model train error: {e}")
    joblib.dump({"model": model, "scaler": scaler}, model_file)
    model_initialized = True
    latest = compute_features(df.iloc[-history_len:])
    return scaler.transform(latest) if latest is not None else None

def compute_trade_amount():
    if len(history_df) < 10:
        return trade_amount
    vol = history_df["close"].pct_change().rolling(10).std().iloc[-1]
    if pd.isna(vol) or vol == 0:
        return trade_amount
    factor = min(1.0, 0.02 / vol)
    return trade_amount * factor
 
def check_balance(side, amount):
    try:
        bal = session.get_wallet_balance(accountType="UNIFIED")
        coins = {c["coin"]: float(c.get("availableToTrade", 0))
                 for c in bal["result"]["list"][0]["coin"]}
        base = symbol.replace("USDT", "")
        if side == "Buy":
            return coins.get("USDT", 0) >= amount
        else:
            return coins.get(base, 0) > 0
    except Exception as e:
        logging.warning(f"Balance check error: {e}")
        return True

def place_order(side, price, amount):
    for _ in range(max_retries):
        try:
            resp = session.place_order(
                category="spot",
                symbol=symbol,
                side=side,
                order_type="Market",
                quote_qty=amount,
                time_in_force="IOC",
            )
            if resp.get("retCode") != 0:
                raise Exception(resp.get("retMsg"))
            order_id = resp["result"]["orderId"]
            for _ in range(5):
                info = session.get_order_history(category="spot", orderId=order_id)
                status = info["result"]["list"][0]["orderStatus"]
                if status == "Filled":
                    msg = f"{side} executed {symbol} at {price} for {amount}"
                    logging.info(msg)
                    send_telegram(msg)
                    with open(trade_file, "a") as f:
                        f.write(f"{time.time()},{side},{price},{amount}\n")
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
    global position_price, trailing_stop_price
    trailing_stop_price = None
    while True:
        try:
            price, volume, bid, ask = get_market_data()
            logging.info(f"Price: {price}")
            features = update_model(price, volume, bid, ask)
            if position_price is not None:
                if trailing_percent > 0:
                    new_stop = price * (1 - trailing_percent / 100)
                    if trailing_stop_price is None or new_stop > trailing_stop_price:
                        trailing_stop_price = new_stop
                sl_level = position_price * (1 - sl_percent / 100)
                if trailing_stop_price:
                    sl_level = max(sl_level, trailing_stop_price)
                if price >= position_price * (1 + tp_percent / 100):
                    amt = compute_trade_amount()
                    if check_balance("Sell", amt) and place_order("Sell", price, amt):
                        send_telegram(f"Take profit at {price}")
                        position_price = None
                        trailing_stop_price = None
                    continue
                if price <= sl_level:
                    amt = compute_trade_amount()
                    if check_balance("Sell", amt) and place_order("Sell", price, amt):
                        send_telegram(f"Stop loss at {price}")
                        position_price = None
                        trailing_stop_price = None
                    continue

            if model_initialized and features is not None:
                prediction = model.predict(features)[0]
                decision = "Buy" if prediction == 1 else "Sell"
            else:
                if len(history_df) >= 2:
                    avg = history_df["close"].mean()
                    decision = "Buy" if price > avg else "Sell"
                else:
                    decision = "Buy" if int(price) % 2 == 0 else "Sell"

            amt = compute_trade_amount()
            if position_price is None and decision == "Buy":
                if check_balance("Buy", amt) and place_order("Buy", price, amt):
                    position_price = price
                    if trailing_percent > 0:
                        trailing_stop_price = price * (1 - trailing_percent / 100)
            elif position_price is not None and decision == "Sell":
                if check_balance("Sell", amt) and place_order("Sell", price, amt):
                    position_price = None
                    trailing_stop_price = None
        except Exception as e:
            logging.error(f"Loop error: {e}")
            send_telegram(f"Bot error: {e}")
        time.sleep(interval)

if __name__ == "__main__":
    trade_loop()
