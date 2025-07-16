import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import ACTIVITIES, MODEL_PATH, FEATURE_COLS
from pcap_processor import pcap_to_dataframe
from feature_extractor import extract_features
from utils import generate_labels  # Funzione da implementare


def train_model():
    # Processa PCAP e estrai features
    df = pcap_to_dataframe()
    features_df = extract_features(df)

    # Genera etichette (DA ADATTARE CON DATI REALI)
    X = features_df[FEATURE_COLS].fillna(0)
    y = generate_labels(X)  # Implementa la logica di labeling

    # Split e addestramento
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Valutazione
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ACTIVITIES.values(),
                yticklabels=ACTIVITIES.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')

    # Salva modello
    joblib.dump(model, MODEL_PATH)
    print(f"Modello salvato in {MODEL_PATH}")

    return model


if __name__ == "__main__":
    train_model()