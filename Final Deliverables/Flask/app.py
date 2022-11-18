import pandas as pd
import numpy as np
import pickle
import os
from flask import Flask, request, render_template
import requests
from dotenv import load_dotenv

# Loading up the values
load_dotenv()

template_dir = os.path.abspath('./templates')
app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def index():
    return render_template("home.html")

@app.route('/predict')
def pred():
    return render_template("upload.html")

@app.route('/predict-value', methods=['POST'])
def predict():
    print("[INFO] loading model...")
    model = pickle.loads(open('../Training/fdemand.pkl', 'rb').read())
    input_features = [float(x) for x in request.form.values()]

    features_value = [np.array(input_features)]
    features_name = ['homepage_featured', 'emailer_for_promotion', 'op_area', 'cuisine', 'city_code', 'region_code', 'category']

    # Access IBM Cloud
    API_KEY = os.environ.get("API_KEY") or None
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
    API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]

    header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

    # NOTE: manually define and pass the array(s) of values to be scored in the next line
    payload_scoring = {"input_data": [{"fields": features_name, "values": [input_features]}]}

    response_scoring = requests.post(
        'https://us-south.ml.cloud.ibm.com/ml/v4/deployments/7f16a51f-5465-40ad-a4d4-85b37976d269/predictions?version=2022-11-18', 
        json=payload_scoring,
    headers={'Authorization': 'Bearer ' + mltoken})
    predicted_value = response_scoring.json()['predictions'][0]['values'][0][0]

    # For Local Deployment
    # prediction = model.predict(features_value)
    # output = prediction[0]
    # print(output)

    return render_template('result.html', no_of_orders = int(predicted_value))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

