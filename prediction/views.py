from django.shortcuts import render
import json
from django.http import JsonResponse

def predict_price(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            crypto = data.get("crypto")
            days = data.get("days")
            algorithm = data.get("algorithm")
            wallet = data.get("wallet", None)  # حالا اختیاری شد!

            if not crypto or not days or not algorithm:
                return JsonResponse({"error": "لطفاً تمام فیلدها را پر کنید."}, status=400)

            predicted_price = 50000  # مقدار فرضی

            response_data = {"price": predicted_price}
            if wallet:
                response_data["wallet"] = wallet  # فقط اگر کیف پول باشد

            return JsonResponse(response_data)

        except json.JSONDecodeError:
            return JsonResponse({"error": "درخواست نامعتبر است."}, status=400)
# Create your views here.
