import os
import numpy as np
import pandas as pd
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Путь к CSV-файлу и корневой папке с аудио
CSV_PATH = "TinySOL_metadata.csv"
AUDIO_ROOT = "."

# Выбираем 6 контрастных инструментов из разных семейств (можно убрать фильтр, чтобы взять все 14)
SELECTED_INSTRUMENTS = [
    "Violin",
    "Cello",
    "Flute",
    "Alto Saxophone",
    "Trumpet in C",
    "Accordion"
]

# 1. Загрузка аудио и извлечение признаков MFCC по таблице метаданных
def load_tinysol_data(csv_path, audio_root, instruments, n_mfcc=20, n_fft=2048, hop_length=512):
    df = pd.read_csv(csv_path)
    df = df[df["Instrument (in full)"].isin(instruments)].reset_index(drop=True)

    X_train, y_train, X_test, y_test = [], [], [], []

    for _, row in df.iterrows():
        file_path = os.path.join(audio_root, row["Path"])
        label = row["Instrument (in full)"]
        fold = row["Fold"]

        try:
            signal, sr = librosa.load(file_path, sr=22050, mono=True, duration=3.0)
            mfcc = librosa.feature.mfcc(
                y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
            )
            features = np.concatenate([np.mean(mfcc, axis=1), np.std(mfcc, axis=1)])

            # Используем официальное разбиение TinySOL: 4-й фолд идет в тест, остальные в обучение
            if fold == 4:
                X_test.append(features)
                y_test.append(label)
            else:
                X_train.append(features)
                y_train.append(label)
        except Exception as e:
            print(f"Ошибка чтения {file_path}: {e}")

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

# 2. Легковесная полносвязная нейросеть (MLP)
class InstrumentMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# 3. Обучение и тестирование для одной конфигурации параметров MFCC
def run_experiment(n_mfcc, n_fft, hop_length, epochs=50):
    torch.manual_seed(42)
    np.random.seed(42)

    X_train, y_train, X_test, y_test = load_tinysol_data(
        CSV_PATH, AUDIO_ROOT, SELECTED_INSTRUMENTS, n_mfcc, n_fft, hop_length
    )

    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_enc, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    model = InstrumentMLP(input_dim=X_train.shape[1], num_classes=len(SELECTED_INSTRUMENTS))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(torch.tensor(X_test, dtype=torch.float32)), dim=1).numpy()

    acc = accuracy_score(y_test_enc, preds)
    f1 = f1_score(y_test_enc, preds, average="macro")
    cm = confusion_matrix(y_test_enc, preds)
    return acc, f1, cm, encoder.classes_, len(X_train), len(X_test)

# 4. Серия замеров для статьи
if __name__ == "__main__":
    configs = [
        (13, 1024, 256),
        (13, 2048, 512),
        (20, 1024, 256),
        (20, 2048, 512),
        (40, 2048, 512),
    ]

    results = []
    best_acc = 0
    best_cm = None
    best_labels = None

    for n_mfcc, n_fft, hop in configs:
        acc, f1, cm, labels, n_tr, n_te = run_experiment(n_mfcc, n_fft, hop)
        results.append((f"MFCC={n_mfcc}\nFFT={n_fft}", acc * 100, f1 * 100))
        print(f"n_mfcc={n_mfcc}, n_fft={n_fft}, hop={hop} (train={n_tr}, test={n_te}) -> Accuracy: {acc*100:.2f}%, F1: {f1*100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_cm = cm
            best_labels = labels

    # График 1: Сравнение конфигураций
    names = [r[0] for r in results]
    accs = [r[1] for r in results]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(names, accs, color="#2b6cb0")
    plt.ylabel("Точность классификации (Accuracy, %)")
    plt.ylim(90, 101)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.3, f"{yval:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig("results_plot.png", dpi=300)

    # График 2: Матрица ошибок для лучшей модели
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", xticklabels=best_labels, yticklabels=best_labels)
    plt.xlabel("Предсказанный инструмент")
    plt.ylabel("Истинный инструмент")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    print("\nГотово! Файлы results_plot.png и confusion_matrix.png сохранены.")