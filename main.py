import time, os, logging
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

telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")

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
    data = session.get_ticker(symbol=symbol)
    return float(data["result"]["list"][0]["lastPrice"])

def place_order(side, price):
    try:
        session.place_order(category="spot", symbol=symbol, side=side,
                            order_type="Market", quote_qty=trade_amount, time_in_force="IOC")
        msg = f"{side} executed {symbol} at {price}"
        logging.info(msg)
        send_telegram(msg)
    except Exception as e:
        logging.error(f"Order error: {e}")
        send_telegram(f"Order error: {e}")

def trade_loop():
    logging.info(f"Bot start: {symbol}, {trade_amount} USDT per trade")
    send_telegram(f"Bot live: trading {symbol}, every {interval//60} min")
    while True:
        try:
            price = get_price()
            logging.info(f"Price: {price}")
            if int(price) % 2 == 0:
                place_order("Buy", price)
            else:
                place_order("Sell", price)
        except Exception as e:
            logging.error(f"Loop error: {e}")
            send_telegram(f"Bot error: {e}")
        time.sleep(interval)

if __name__ == "__main__":
    trade_loop()
