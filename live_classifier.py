import pyshark
import time
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from collections import deque
from config import USER_MAC, MODEL_PATH, ACTIVITIES, WINDOW_SIZE, FEATURE_COLS
from feature_extractor import extract_features


class LiveClassifier:
    def __init__(self, interface='wlan0mon'):
        self.model = joblib.load(MODEL_PATH)
        self.interface = interface
        self.buffer = deque(maxlen=10000)
        self.last_time = time.time()

    def packet_handler(self, pkt):
        try:
            if hasattr(pkt, 'wlan'):
                src = pkt.wlan.sa
                dst = pkt.wlan.da
                if USER_MAC not in [src, dst]:
                    return

                direction = 'up' if src == USER_MAC else 'down'
                self.buffer.append({
                    'timestamp': pkt.sniff_time.timestamp(),
                    'packet_size': int(pkt.length),
                    'direction': direction
                })

                # Processa ogni W secondi
                if time.time() - self.last_time >= WINDOW_SIZE:
                    self.process_window()
                    self.last_time = time.time()

        except AttributeError:
            pass

    def process_window(self):
        if not self.buffer:
            return

        df = pd.DataFrame(self.buffer)
        features = extract_features(df)[FEATURE_COLS].fillna(0).iloc[-1:]

        if not features.empty:
            activity = self.model.predict(features)[0]
            print(f"[{datetime.now()}] Attività: {ACTIVITIES[activity]}")

        # Pulisci buffer
        self.buffer.clear()

    def start(self):
        print(f"Avvio sniffing su {self.interface}...")
        capture = pyshark.LiveCapture(
            interface=self.interface,
            display_filter=f"wlan.addr == {USER_MAC}"
        )
        capture.apply_on_packets(self.packet_handler)


if __name__ == "__main__":
    classifier = LiveClassifier()
    classifier.start()