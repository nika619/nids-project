import socket
import time

print("Simulating DoS traffic (SYN flood to localhost)...")
for i in range(100):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.1)
        s.connect(('127.0.0.1', 5000))
        s.close()
    except:
        pass
    time.sleep(0.05)
    print(f"Packet {i+1} sent")

print("Simulation complete.")
