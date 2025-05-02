from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

model = joblib.load('fraud_model.pkl') 
scaler = joblib.load('scaler.pkl')
features = joblib.load('features.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    
    # Scale and encode
    df['Amount'] = scaler.transform(df[['Amount']])
    df = pd.get_dummies(df)
    df = df.reindex(columns=features, fill_value=0)
    
    probas = model.predict_proba(df)[0]
    prediction = int(np.argmax(probas))
    
    return jsonify({
        'fraud': prediction,
        'probabilities': {
            'Not Fraud': round(probas[0] * 100, 2),
            'Fraud': round(probas[1] * 100, 2)
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
