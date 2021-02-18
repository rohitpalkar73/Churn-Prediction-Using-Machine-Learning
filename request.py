import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'balance':159660, 'Tenure':2, 'Age':40,  'gender':0,  'Enter number of products':3,  'enter salary':113931, 'is active member':0, 'has credit card':1, 'credit score':502, 'has exited':1})

print(r.json())