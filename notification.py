# /root/backend/notification.py

import firebase_admin
from firebase_admin import credentials, messaging
from flask import request, jsonify
from crypto_predictor import app  # فرض می‌کنیم فایل اصلی API را import می‌کنیم

# بارگذاری اعتبارنامه Firebase (اطمینان حاصل کنید فایل firebase_credentials.json در پوشه backend قرار دارد)
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)

notification_tokens = set()

@app.route("/subscribe", methods=["POST"])
def subscribe():
    data = request.json
    token = data.get("token")
    if token:
        notification_tokens.add(token)
        return jsonify({"status": "subscribed"}), 200
    return jsonify({"error": "Missing token"}), 400

@app.route("/notify", methods=["POST"])
def notify():
    data = request.json
    title = data.get("title", "Crypto Alert")
    body = data.get("body", "Your prediction is ready!")
    responses = []
    for token in notification_tokens:
        message = messaging.Message(
            notification=messaging.Notification(title=title, body=body),
            token=token
        )
        response = messaging.send(message)
        responses.append(response)
    return jsonify({"responses": responses})
