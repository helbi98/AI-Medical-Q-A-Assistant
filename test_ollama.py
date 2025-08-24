import requests

url = "http://localhost:11434/api/generate"
payload = {"model": "gemma3:1b", "prompt": "Hello, what is COPD?", "stream": False}

resp = requests.post(url, json=payload)
resp.raise_for_status()
print(resp.json())
