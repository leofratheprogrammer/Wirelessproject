import pyshark
import pandas as pd
from config import USER_MAC, PCAP_PATH


def pcap_to_dataframe():
    """Converte pcapng in DataFrame strutturato"""
    packets = []
    cap = pyshark.FileCapture(PCAP_PATH, display_filter=f"wlan.addr == {USER_MAC}")

    for pkt in cap:
        try:
            # Estrai informazioni strato Wi-Fi
            if hasattr(pkt, 'wlan'):
                src = pkt.wlan.sa
                dst = pkt.wlan.da
                size = int(pkt.length)

                packets.append({
                    'timestamp': pkt.sniff_time.timestamp(),
                    'src_mac': src,
                    'dst_mac': dst,
                    'packet_size': size
                })
        except AttributeError:
            continue

    df = pd.DataFrame(packets)

    # Calcola direzione
    df['direction'] = df['src_mac'].apply(
        lambda x: 'up' if x == USER_MAC else 'down'
    )

    return df[['timestamp', 'packet_size', 'direction']]