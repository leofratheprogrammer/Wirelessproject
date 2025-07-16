import numpy as np
import pandas as pd
from config import WINDOW_SIZE


def extract_features(df):
    """Calcola features statistiche per finestre temporali"""
    # Ordina per timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)

    # Calcola inter-arrival time
    df['iat'] = df['timestamp'].diff().fillna(0)

    # Raggruppa per finestre temporali
    features_list = []
    for _, group in df.groupby(pd.cut(df['timestamp'],
                                      bins=np.arange(df['timestamp'].min(),
                                                     df['timestamp'].max() + WINDOW_SIZE,
                                                     WINDOW_SIZE))):
        if group.empty:
            continue

        features = {
            'start_time': group['timestamp'].min(),
            'end_time': group['timestamp'].max()
        }

        # Conteggi pacchetti
        dir_counts = group['direction'].value_counts()
        features['num_up'] = dir_counts.get('up', 0)
        features['num_down'] = dir_counts.get('down', 0)

        # Statistiche dimensioni
        for direction in ['up', 'down']:
            dir_data = group[group['direction'] == direction]
            sizes = dir_data['packet_size']

            features[f'bytes_{direction}'] = sizes.sum()
            features[f'avg_size_{direction}'] = sizes.mean() if not sizes.empty else 0
            features[f'var_size_{direction}'] = sizes.var() if len(sizes) > 1 else 0

        # Statistiche inter-arrival
        iats = group['iat']
        features['avg_iat'] = iats.mean()
        features['var_iat'] = iats.var() if len(iats) > 1 else 0

        # Ratio upstream/downstream
        features['up_down_ratio'] = features['num_up'] / features['num_down'] if features['num_down'] > 0 else 0

        features_list.append(features)

    return pd.DataFrame(features_list)