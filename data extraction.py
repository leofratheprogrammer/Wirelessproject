import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# Parametri
USER_MAC = "AA:BB:CC:DD:EE:FF"  # Sostituisci con il MAC target
WINDOW_SIZE = 5  # Secondi per finestra temporale

# Carica i dati
df = pd.read_csv('traffic.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filtra solo traffico dell'utente target
user_traffic = df[(df['src_mac'] == USER_MAC) | (df['dst_mac'] == USER_MAC)]

# Calcola direzione
user_traffic['direction'] = np.where(
    user_traffic['src_mac'] == USER_MAC,
    'up',
    'down'
)


# Funzione per estrarre features per finestra
def extract_features(window):
    features = {}

    # Conteggi pacchetti
    features['num_up'] = (window['direction'] == 'up').sum()
    features['num_down'] = (window['direction'] == 'down').sum()

    # Statistiche dimensione pacchetti
    features['avg_size_up'] = window[window['direction'] == 'up']['packet_size'].mean()
    features['var_size_up'] = window[window['direction'] == 'up']['packet_size'].var()
    features['avg_size_down'] = window[window['direction'] == 'down']['packet_size'].mean()
    features['var_size_down'] = window[window['direction'] == 'down']['packet_size'].var()

    # Statistiche inter-arrival (calcola differenze temporali)
    window = window.sort_values('timestamp')
    window['iat'] = window['timestamp'].diff().dt.total_seconds()
    features['avg_iat'] = window['iat'].mean()
    features['var_iat'] = window['iat'].var()

    return pd.Series(features)


# Estrai features ogni W secondi
features = user_traffic.groupby(
    pd.Grouper(key='timestamp', freq=f'{WINDOW_SIZE}S')
).apply(extract_features).fillna(0)