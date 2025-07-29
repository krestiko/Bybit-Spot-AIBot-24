import time
import os
import logging
import numpy as np
import requests
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier
import pandas as pd
np.NaN = np.nan  # compatibility for pandas_ta
import pandas_ta as ta
import joblib
from pybit.unified_trading import HTTP
from dotenv import load_dotenv
import yaml

load_dotenv()
config_path = os.getenv("CONFIG_FILE", "config.yaml")
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
else:
    config = {}
log_file = os.getenv("LOG_FILE", "bot.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
)

api_key = os.getenv("BYBIT_API_KEY")
api_secret = os.getenv("BYBIT_API_SECRET")
symbol = os.getenv("SYMBOL", "BTCUSDT")
trade_amount = float(os.getenv("TRADE_AMOUNT_USDT", 10))
tp_percent = float(config.get("trade", {}).get("tp_percent", os.getenv("TP_PERCENT", 1.5)))
sl_percent = float(config.get("trade", {}).get("sl_percent", os.getenv("SL_PERCENT", 1.0)))
interval = int(os.getenv("INTERVAL", 5)) * 60
max_retries = int(os.getenv("MAX_RETRIES", 3))

# length of price history to use for learning
history_len = int(os.getenv("HISTORY_LENGTH", 100))
price_file = os.getenv("PRICE_HISTORY_FILE", "price_history.csv")
trade_file = os.getenv("TRADE_HISTORY_FILE", "trade_history.csv")
trailing_percent = float(os.getenv("TRAILING_PERCENT", 0))
model_type = config.get("trade", {}).get("model_type", os.getenv("MODEL_TYPE", "gb"))
daily_loss_limit = float(os.getenv("DAILY_STOP_LOSS", 0))
daily_profit_limit = float(os.getenv("DAILY_TAKE_PROFIT", 0))

indicators = config.get("trade", {}).get(
    "indicators",
    [
        "rsi",
        "macd",
        "bb",
        "sma",
        "ema",
        "adx",
        "stoch",
        "obv",
    ],
)

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
    if model_type == "xgb":
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    elif model_type == "gb":
        model = GradientBoostingClassifier()
    else:
        model = SGDClassifier(loss="log_loss")
    model_initialized = False

# price and amount of the current position; None means no position
position_price = None
position_amount = 0.0
daily_pnl = 0.0
daily_date = time.strftime("%Y-%m-%d")

def send_telegram(msg):
    if not telegram_token or not telegram_chat_id: return
    try:
        requests.post(f"https://api.telegram.org/bot{telegram_token}/sendMessage",
                      data={"chat_id": telegram_chat_id, "text": msg})
    except Exception as e:
        logging.warning(f"Telegram error: {e}")

def fetch_news_sentiment():
    token = os.getenv("CRYPTO_NEWS_TOKEN")
    if not token:
        return 0.0
    try:
        resp = requests.get(
            "https://cryptopanic.com/api/v1/posts/",
            params={"auth_token": token, "kind": "news", "public": "true"},
            timeout=10,
        )
        data = resp.json()
        posts = data.get("results", [])[:10]
        score = 0.0
        for p in posts:
            if p.get("positive_votes", 0) >= p.get("negative_votes", 0):
                score += 1
            else:
                score -= 1
        return score / max(len(posts), 1)
    except Exception as e:
        logging.warning(f"News fetch error: {e}")
        return 0.0

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
    feats = []
    required = []
    if "rsi" in indicators:
        df["rsi"] = ta.rsi(df["close"], length=14)
        required.append("rsi")
    if "macd" in indicators:
        macd = ta.macd(df["close"])
        df["macd"] = macd["MACD_12_26_9"]
        required.append("macd")
    if "bb" in indicators:
        bb = ta.bbands(df["close"], length=5)
        df["bb_upper"] = bb["BBU_5_2.0"]
        df["bb_lower"] = bb["BBL_5_2.0"]
        required.extend(["bb_upper", "bb_lower"])
    if "sma" in indicators:
        df["sma"] = ta.sma(df["close"], length=10)
        required.append("sma")
    if "ema" in indicators:
        df["ema"] = ta.ema(df["close"], length=10)
        required.append("ema")
    if "adx" in indicators:
        adx = ta.adx(df["close"], length=14)
        df["adx"] = adx["ADX_14"]
        required.append("adx")
    if "stoch" in indicators:
        stoch = ta.stoch(df["close"])
        df["stoch_k"] = stoch["STOCHk_14_3_3"]
        df["stoch_d"] = stoch["STOCHd_14_3_3"]
        required.extend(["stoch_k", "stoch_d"])
    if "obv" in indicators:
        df["obv"] = ta.obv(df["close"], df["volume"])
        required.append("obv")
    df["bid_ask_ratio"] = (df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"])
    required.append("bid_ask_ratio")
    if "news" in indicators:
        df["sentiment"] = fetch_news_sentiment()
        required.append("sentiment")
    row = df.iloc[-1]
    if row[required].isna().any():
        return None
    feats = row[required + ["volume"]].values
    return np.array(feats, dtype=float).reshape(1, -1)

def append_market_data(price, volume, bid, ask):
    global history_df
    history_df.loc[len(history_df)] = [price, volume, bid, ask]
    if len(history_df) > history_len + 50:
        history_df = history_df.iloc[-(history_len + 50):]
    history_df.to_csv(price_file, index=False)

def update_model(price, volume, bid, ask):
    global model_initialized, model
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
    if len(features_list) > history_len:
        features_list.pop(0)
        labels_list.pop(0)
    X = np.array(features_list)
    y = np.array(labels_list)
    scaler.fit(X)
    Xs = scaler.transform(X)
    try:
        if len(y) >= 20:
            if model_type == "xgb":
                params = {"n_estimators": [50, 100], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
            elif model_type == "gb":
                params = {"n_estimators": [50, 100], "max_depth": [3, 5]}
            else:
                params = {"alpha": [0.0001, 0.001]}
            grid = GridSearchCV(model, params, cv=3, n_jobs=-1)
            grid.fit(Xs, y)
            model = grid.best_estimator_
            score = grid.best_score_
        else:
            model.fit(Xs, y)
            score = cross_val_score(model, Xs, y, cv=3).mean() if len(y) >= 3 else None
        if score is not None:
            logging.info(f"CV score: {score:.3f}")
    except Exception as e:
        logging.warning(f"Model train error: {e}")
    if model_initialized and hasattr(model, "partial_fit"):
        try:
            model.partial_fit(scaler.transform(features), [label], classes=np.array([0, 1]))
        except Exception as e:
            logging.warning(f"Partial fit error: {e}")
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

def reset_daily_pnl():
    global daily_pnl, daily_date
    today = time.strftime("%Y-%m-%d")
    if today != daily_date:
        daily_date = today
        daily_pnl = 0.0

def update_pnl(profit):
    global daily_pnl
    daily_pnl += profit

def log_trade(side, price, qty, pnl=0.0, tp=None, sl=None):
    header = not os.path.exists(trade_file)
    with open(trade_file, "a") as f:
        if header:
            f.write("time,side,price,qty,pnl,tp,sl\n")
        f.write(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')},{side},{price},{qty},{pnl},{tp},{sl}\n"
        )


simulate = not (api_key and api_secret)
use_websocket = os.getenv("USE_WEBSOCKET", "0") == "1"
last_price = None
position_side = None  # 'long' or 'short'
trailing_price = None

def place_order(side, qty, tp=None, sl=None):
    if simulate:
        logging.info(f"Simulated order: {side} {qty}")
        return {"price": last_price, "qty": qty}
    for _ in range(max_retries):
        try:
            return session.place_order(
                category="spot",
                symbol=symbol,
                side="Buy" if side.lower() == "buy" else "Sell",
                orderType="Market",
                qty=qty,
                takeProfit=tp,
                stopLoss=sl,
            )
        except Exception as e:
            logging.warning(f"Order error: {e}")
            time.sleep(1)
    raise RuntimeError("Failed to place order")


def open_long(price):
    global position_price, position_amount, position_side, trailing_price
    qty = compute_trade_amount() / price
    place_order("buy", qty, tp=price * (1 + tp_percent / 100), sl=price * (1 - sl_percent / 100))
    position_price = price
    position_amount = qty
    position_side = "long"
    trailing_price = price * (1 - trailing_percent / 100) if trailing_percent else None
    log_trade("buy", price, qty, tp=tp_percent, sl=sl_percent)
    send_telegram(f"Opened long {qty:.6f} {symbol} @ {price}")


def open_short(price):
    global position_price, position_amount, position_side, trailing_price
    qty = compute_trade_amount() / price
    place_order("sell", qty, tp=price * (1 - tp_percent / 100), sl=price * (1 + sl_percent / 100))
    position_price = price
    position_amount = -qty
    position_side = "short"
    trailing_price = price * (1 + trailing_percent / 100) if trailing_percent else None
    log_trade("sell", price, qty, tp=tp_percent, sl=sl_percent)
    send_telegram(f"Opened short {qty:.6f} {symbol} @ {price}")


def close_position(price):
    global position_price, position_amount, position_side, trailing_price
    if position_amount == 0:
        return
    side = "sell" if position_amount > 0 else "buy"
    qty = abs(position_amount)
    place_order(side, qty)
    pnl = (price - position_price) * position_amount
    update_pnl(pnl)
    log_trade("close", price, qty, pnl)
    send_telegram(f"Closed position {side} pnl={pnl:.2f}")
    position_amount = 0
    position_price = None
    position_side = None
    trailing_price = None


def handle_trailing(price):
    global trailing_price
    if trailing_price is None:
        return False
    if position_side == "long":
        if price > position_price and price * (1 - trailing_percent / 100) > trailing_price:
            trailing_price = price * (1 - trailing_percent / 100)
        if price <= trailing_price:
            return True
    else:
        if price < position_price and price * (1 + trailing_percent / 100) < trailing_price:
            trailing_price = price * (1 + trailing_percent / 100)
        if price >= trailing_price:
            return True
    return False


def trade_cycle():
    global last_price
    if use_websocket:
        from pybit.unified_trading import WebSocket

        def _cb(msg):
            global last_price
            data = msg.get("data")
            if isinstance(data, list):
                data = data[0]
            lp = data.get("lastPrice") or data.get("lp")
            if lp:
                last_price = float(lp)

        ws = WebSocket("spot")
        ws.ticker_stream(symbol, _cb)

    while True:
        reset_daily_pnl()
        if daily_loss_limit and daily_pnl <= -daily_loss_limit:
            logging.warning("Daily loss limit reached")
            break
        if daily_profit_limit and daily_pnl >= daily_profit_limit:
            logging.info("Daily profit target reached")
            break
        if last_price is None:
            price, vol, bid, ask = get_market_data()
        else:
            price = last_price
            vol = 0
            bid = ask = 0
        features = update_model(price, vol, bid, ask)
        if features is None:
            time.sleep(interval)
            continue
        prob = model.predict_proba(features)[0][1]

        if position_amount > 0:  # long
            if price >= position_price * (1 + tp_percent / 100) or price <= position_price * (1 - sl_percent / 100) or handle_trailing(price):
                close_position(price)
        elif position_amount < 0:  # short
            if price <= position_price * (1 - tp_percent / 100) or price >= position_price * (1 + sl_percent / 100) or handle_trailing(price):
                close_position(price)
        if position_amount == 0:
            if prob > 0.55:
                open_long(price)
            elif prob < 0.45:
                open_short(price)
        time.sleep(interval)


if __name__ == "__main__":
    trade_cycle()
