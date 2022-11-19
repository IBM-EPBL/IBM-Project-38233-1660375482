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

    input_features = [float(x) for x in request.form.values()]

    features_name = ['homepage_featured', 'emailer_for_promotion', 'op_area', 'cuisine', 'city_code', 'region_code', 'category']

    predicted_value = predict_values(features_name, input_features)['predictions'][0]['values'][0][0]

    return render_template('result.html', no_of_orders = int(predicted_value))

def predict_values(feature_names, feature_values):
    # Access IBM Cloud
    # Get API KEY 
    API_KEY = os.environ.get("API_KEY") or None
    # Get MLToken
    token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
    API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
    mltoken = token_response.json()["access_token"]

    # Prepare Payload
    payload = {"input_data": [{"fields": feature_names, "values": [feature_values]}]}

    response = requests.post(
        'https://us-south.ml.cloud.ibm.com/ml/v4/deployments/7f16a51f-5465-40ad-a4d4-85b37976d269/predictions?version=2022-11-18', 
        json=payload,
    headers={'Authorization': 'Bearer ' + mltoken})

    print(response.json())
    return response.json()

    # ----------------- For Local Deployment -------------------------------
    # print("[INFO] loading model...")
    # model = pickle.loads(open('../Training/fdemand.pkl', 'rb').read())
    # features_value = [np.array(input_features)]
    # prediction = model.predict(features_value)
    # output = prediction[0]
    # print(output)
    # return output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

