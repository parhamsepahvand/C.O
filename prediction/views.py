from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

@csrf_exempt
def predict_price(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            crypto = data.get("crypto")
            days = data.get("days")
            algorithm = data.get("algorithm")

            if not crypto or not days or not algorithm:
                return JsonResponse({"error": "Missing required fields."}, status=400)

            predicted_price = 50000  # مقدار تستی
            return JsonResponse({"predicted_price": predicted_price, "algorithm": algorithm})

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid request format."}, status=400)

    return JsonResponse({"error": "Only POST requests are allowed."}, status=405)
