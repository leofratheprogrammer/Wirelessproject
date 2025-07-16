import pandas as pd
from config import DATA_PATH, USER_MAC


def load_and_preprocess_data():
    """Carica e prepara i dati grezzi"""
    df = pd.read_csv(DATA_PATH)

    # Conversione timestamp e filtro MAC
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    filtered = df[(df['src_mac'] == USER_MAC) | (df['dst_mac'] == USER_MAC)].copy()

    # Calcolo direzione pacchetti
    filtered['direction'] = filtered.apply(
        lambda x: 'up' if x['src_mac'] == USER_MAC else 'down',
        axis=1
    )

    return filtered[['timestamp', 'packet_size', 'direction']]