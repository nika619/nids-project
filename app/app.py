from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os
import datetime

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
model         = joblib.load(os.path.join(BASE, '../models/model.pkl'))
scaler        = joblib.load(os.path.join(BASE, '../models/scaler.pkl'))
label_encoder = joblib.load(os.path.join(BASE, '../models/label_encoder.pkl'))

logs = []

@app.route('/')
def home():
    return render_template('index.html', prediction=None, logs=logs[-10:])

# Manual classify from form
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form.get(f'f{i}', 0)) for i in range(41)]
    features = np.array(features).reshape(1, -1)
    features = scaler.transform(features)
    pred_num   = model.predict(features)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]
    color = 'lime' if pred_label == 'normal' else 'red'

    logs.append({
        'time':   datetime.datetime.now().strftime("%H:%M:%S"),
        'result': pred_label,
        'color':  color,
        'source': 'Manual'
    })
    return render_template('index.html', prediction=pred_label, color=color, logs=logs[-10:])

# Auto classify from live_detect.py
@app.route('/api/detect', methods=['POST'])
def api_detect():
    data     = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    features = scaler.transform(features)
    pred_num   = model.predict(features)[0]
    pred_label = label_encoder.inverse_transform([pred_num])[0]
    color = 'lime' if pred_label == 'normal' else 'red'

    logs.append({
        'time':   datetime.datetime.now().strftime("%H:%M:%S"),
        'result': pred_label,
        'color':  color,
        'source': data.get('src_ip', 'Unknown') + ' -> ' + data.get('dst_ip', 'Unknown')
    })
    return jsonify({'prediction': pred_label, 'color': color})

# Get latest logs as JSON (for auto-refresh)
@app.route('/api/logs')
def get_logs():
    return jsonify(logs[-10:])

if __name__ == '__main__':
    app.run(debug=True)
