# Parametri configurabili
WINDOW_SIZE = 5  # Secondi per finestra temporale
USER_MAC = "2e:e6:40:80:45:03"  # MAC address target
MODEL_PATH = "wifi_classifier_model.pkl"
PCAP_PATH = "traffic.pcapng"
ACTIVITIES = {
    0: "idle",
    1: "web browsing",
    2: "YouTube streaming"
}
FEATURE_COLS = [
    'num_up', 'num_down', 'bytes_up', 'bytes_down',
    'avg_size_up', 'var_size_up', 'avg_size_down', 'var_size_down',
    'avg_iat', 'var_iat', 'up_down_ratio'
]