import ccxt
from cryptography.fernet import Fernet
from django.conf import settings

class BinanceAPI:
    def __init__(self):
        self.cipher = Fernet(settings.ENCRYPTION_KEY)
        self.client = ccxt.binance({
            'apiKey': self._decrypt(settings.BINANCE_API_KEY),
            'secret': self._decrypt(settings.BINANCE_SECRET_KEY),
            'options': {'defaultType': 'future'}
        })
    
    def _decrypt(self, encrypted):
        return self.cipher.decrypt(encrypted.encode()).decode()
    
    def create_order(self, symbol, side, amount):
        return self.client.create_market_order(symbol, side, amount)
