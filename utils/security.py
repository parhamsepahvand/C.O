from cryptography.fernet import Fernet
from django.conf import settings

class SecurityVault:
    def __init__(self):
        self.cipher = Fernet(settings.ENCRYPTION_KEY)
    
    def encrypt(self, text):
        return self.cipher.encrypt(text.encode()).decode()
    
    def decrypt(self, encrypted):
        return self.cipher.decrypt(encrypted.encode()).decode()
