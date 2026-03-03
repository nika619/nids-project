import warnings
warnings.filterwarnings('ignore')

from scapy.all import sniff, IP, TCP, UDP
import joblib
import numpy as np
import datetime
import requests

model         = joblib.load('models/model.pkl')
scaler        = joblib.load('models/scaler.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

FLASK_URL = 'http://127.0.0.1:5000/api/detect'

def extract_features(packet):
    features = [0] * 41
    if IP in packet:
        features[4] = len(packet[IP])
        features[5] = 0
    if TCP in packet:
        features[2] = 1
        features[22] = packet[TCP].window
        flags = packet[TCP].flags
        if flags & 0x02:
            features[24] = 1.0
            features[25] = 1.0
    elif UDP in packet:
        features[2] = 2
    return features

def classify_packet(packet):
    if not IP in packet:
        return

    features = extract_features(packet)
    src_ip   = packet[IP].src
    dst_ip   = packet[IP].dst
    time     = datetime.datetime.now().strftime("%H:%M:%S")

    # Send to Flask dashboard
    try:
        response = requests.post(FLASK_URL, json={
            'features': features,
            'src_ip':   src_ip,
            'dst_ip':   dst_ip
        })
        result = response.json()['prediction']
        if result != 'normal':
            print(f"[{time}] ALERT! {result} | {src_ip} -> {dst_ip}")
        else:
            print(f"[{time}] Normal | {src_ip} -> {dst_ip}")
    except Exception as e:
        print(f"[{time}] Flask not running: {e}")

print("Starting live packet capture... Press Ctrl+C to stop")
sniff(prn=classify_packet, store=0, count=0)
