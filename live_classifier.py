import time
import joblib
import pandas as pd
from config import MODEL_PATH, WINDOW_SIZE, ACTIVITIES
from feature_extractor import extract_features
from data_loader import load_and_preprocess_data


def simulate_live_classification():
    """Simula classificazione in tempo reale su dati CSV"""
    # Caricamento modello
    model = joblib.load(MODEL_PATH)
    df = load_and_preprocess_data()

    # Finestra temporale scorrevole
    start_time = df['timestamp'].min()
    end_time = start_time + pd.Timedelta(seconds=WINDOW_SIZE)

    while end_time <= df['timestamp'].max():
        # Estrazione finestra corrente
        window = df[(df['timestamp'] >= start_time) &
                    (df['timestamp'] < end_time)]

        if not window.empty:
            # Estrazione features e predizione
            features = extract_features(window).to_frame().T.fillna(0)
            activity_id = model.predict(features)[0]

            print(f"[{start_time} - {end_time}] Attività rilevata: {ACTIVITIES[activity_id]}")

        # Scorrimento finestra
        start_time = end_time
        end_time = start_time + pd.Timedelta(seconds=WINDOW_SIZE)
        time.sleep(1)  # Simula tempo reale


if __name__ == "__main__":
    simulate_live_classification()