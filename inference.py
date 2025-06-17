import requests

URL = "http://localhost:8000/recommend"
payload = {
    "titles": ["The Matrix", "Inception"]
}

res = requests.post(URL, json=payload)
print("Status:", res.status_code)
print("Hasil:", res.json())
