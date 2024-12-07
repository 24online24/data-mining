import json

try:
    with open('secrets.json') as f:
        secrets = json.load(f)
except Exception as e:
    print('Error: ', e)

print(secrets['apiKey'])
print(secrets['secretKey'])
