import requests

response = requests.post("http://127.0.0.1:9000/ingest")

print(response.status_code)
print(response.json())
