import pandas as pd


def extract_features(group):
    """Calcola le features statistiche per una finestra temporale"""
    features = {}
    window = group.sort_values('timestamp')

    # Conteggio pacchetti
    directions = window['direction'].value_counts()
    features['num_up'] = directions.get('up', 0)
    features['num_down'] = directions.get('down', 0)

    # Statistiche dimensioni
    for dir in ['up', 'down']:
        sizes = window[window['direction'] == dir]['packet_size']
        features[f'avg_size_{dir}'] = sizes.mean() if not sizes.empty else 0
        features[f'var_size_{dir}'] = sizes.var() if len(sizes) > 1 else 0

    # Inter-arrival times
    time_diffs = window['timestamp'].diff().dt.total_seconds().dropna()
    features['avg_iat'] = time_diffs.mean() if not time_diffs.empty else 0
    features['var_iat'] = time_diffs.var() if len(time_diffs) > 1 else 0

    return pd.Series(features)