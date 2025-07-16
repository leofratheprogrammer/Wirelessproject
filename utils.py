import numpy as np
from config import ACTIVITIES


def generate_labels(features_df):
    """
    Genera etichette sintetiche basate su pattern di traffico
    (Sostituire con etichette reali in produzione)
    """
    labels = []
    for _, row in features_df.iterrows():
        # YouTube: alto downstream, bassa varianza dimensioni
        if row['bytes_down'] > 500000 and row['var_size_down'] < 1000:
            labels.append(2)  # YouTube

        # Web browsing: traffico bilanciato
        elif 50 < row['num_down'] < 500 and 20 < row['num_up'] < 200:
            labels.append(1)  # Web

        # Idle: poco traffico
        else:
            labels.append(0)  # Idle

    return np.array(labels)