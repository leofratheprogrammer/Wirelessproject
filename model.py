import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from config import ACTIVITIES, MODEL_PATH
from data_loader import load_and_preprocess_data
from feature_extractor import extract_features


def train_and_evaluate():
    """Addestra il modello e valuta le performance"""
    # Caricamento e preparazione dati
    df = load_and_preprocess_data()

    # Estrazione features (simulazione etichette)
    windows = df.groupby(pd.Grouper(key='timestamp', freq=f'{WINDOW_SIZE}S'))
    features = windows.apply(extract_features).fillna(0)

    # Generazione etichette sintetiche (sostituire con dati reali)
    labels = np.random.choice(list(ACTIVITIES.keys()), size=len(features))

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.3, random_state=42
    )

    # Addestramento modello
    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Valutazione
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Confusion matrix
    cm = confusion_matrix(
        [ACTIVITIES[y] for y in y_test],
        [ACTIVITIES[y] for y in y_pred],
        labels=list(ACTIVITIES.values())
    )

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=ACTIVITIES.values(),
                yticklabels=ACTIVITIES.values())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Salvataggio modello
    joblib.dump(model, MODEL_PATH)
    print(f"Modello salvato in {MODEL_PATH}")


if __name__ == "__main__":
    train_and_evaluate()