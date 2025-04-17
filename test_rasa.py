import requests

response = requests.post(
    "http://localhost:5005/webhooks/rest/webhook",
    json={"sender": "test_user", "message": "hello"}
)

print(response.json())
