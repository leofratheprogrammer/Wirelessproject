import numpy as np
from config import ACTIVITIES

def generate_synthetic_labels(features):
    """Genera etichette sintetiche per il training (da sostituire con dati reali)"""
    # Logica basata su pattern noti:
    # - Molti pacchetti down + piccole variazioni = streaming
    # - Pacchetti up/down bilanciati = web browsing
    # - Pochi pacchetti = idle
    labels = []
    for _, row in features.iterrows():
        if row['num_down'] > 50 and row['var_size_down'] < 1000:
            labels.append(2)  # YouTube
        elif 10 < row['num_down'] < 50 and 5 < row['num_up'] < 30:
            labels.append(1)  # Web
        else:
            labels.append(0)  # Idle
    return np.array(labels)