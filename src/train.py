# Bloque de uso: ejecutar desde la raíz del proyecto con:
# python src/train.py
# Dependencias: pandas, numpy, scipy, scikit-learn, torch, matplotlib, joblib
# Este script carga los artefactos del Paso 4, entrena un clasificador de sentimiento en PyTorch
# y guarda checkpoints, métricas y reportes en artifacts/ y data/processed/.

import os
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse
from scipy.sparse import load_npz
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Forzar salida UTF-8 en Windows para evitar problemas de consola.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Rutas base del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Rutas de entrada y salida.
X_PATH = ARTIFACTS_DIR / 'X_tfidf.npz'
Y_PATH = ARTIFACTS_DIR / 'y.npy'
CLASS_WEIGHTS_PATH = ARTIFACTS_DIR / 'class_weights.npy'
VECTORIZER_PATH = ARTIFACTS_DIR / 'tfidf_vectorizer.pkl'
TRAIN_INFO_PATH = ARTIFACTS_DIR / 'train_val_test_split_info.txt'
TRAINING_HISTORY_PATH = ARTIFACTS_DIR / 'training_history.csv'
TRAINING_CONFIG_PATH = ARTIFACTS_DIR / 'training_config.txt'
BEST_MODEL_PATH = ARTIFACTS_DIR / 'best_model.pt'
CLASSIFICATION_REPORT_PATH = ARTIFACTS_DIR / 'classification_report.txt'
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / 'confusion_matrix.png'
TEST_RESULTS_PATH = ARTIFACTS_DIR / 'test_results.txt'
MISCLASSIFIED_PATH = ARTIFACTS_DIR / 'misclassified_examples.csv'
MEMORY_STRATEGY_PATH = ARTIFACTS_DIR / 'memory_strategy.txt'
TRAIN_PT = PROCESSED_DIR / 'train.pt'
VAL_PT = PROCESSED_DIR / 'val.pt'
TEST_PT = PROCESSED_DIR / 'test.pt'

# Parámetros reproducibles.
RANDOM_STATE = 42
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 5
INPUT_DIM = None
N_CLASSES = 2

# Dispositivo de ejecución.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed: int = 42) -> None:
    """Fijar la semilla para reproducibilidad en numpy, random y torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_or_prepare_split() -> tuple[sparse.spmatrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cargar los splits si existen o recrearlos con train/val/test estratificados."""
    if X_PATH.exists() and Y_PATH.exists():
        X = sparse.load_npz(X_PATH)
        y = np.load(Y_PATH)
        if (PROCESSED_DIR / 'train.pt').exists() and (PROCESSED_DIR / 'val.pt').exists() and (PROCESSED_DIR / 'test.pt').exists():
            # Si los archivos .pt existen, se cargan directamente.
            train_data = torch.load(TRAIN_PT, map_location=device)
            val_data = torch.load(VAL_PT, map_location=device)
            test_data = torch.load(TEST_PT, map_location=device)
            X_train = train_data['X'].cpu().to_dense().numpy().astype(np.float32)
            y_train = train_data['y'].cpu().numpy()
            X_val = val_data['X'].cpu().to_dense().numpy().astype(np.float32)
            y_val = val_data['y'].cpu().numpy()
            X_test = test_data['X'].cpu().to_dense().numpy().astype(np.float32)
            y_test = test_data['y'].cpu().numpy()
            return X_train, y_train, X_val, y_val, X_test, y_test

        # Si no hay .pt, se rehace el split estratificado usando la matriz TF-IDF.
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=RANDOM_STATE
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    raise FileNotFoundError('No se encontraron artifacts/X_tfidf.npz y artifacts/y.npy')


def to_dense_if_possible(X: sparse.spmatrix) -> np.ndarray:
    """Convertir a dense si cabe en memoria; si no, documentar estrategia sparse para batch."""
    try:
        dense = X.toarray().astype(np.float32)
        return dense
    except MemoryError:
        with MEMORY_STRATEGY_PATH.open('w', encoding='utf-8') as f:
            f.write('La matriz TF-IDF es demasiado grande para densificar. Se usa Dataset sparse->dense por batch durante el entrenamiento.\n')
        return X.astype(np.float32)


class SparseRowDataset(Dataset):
    """Dataset que convierte filas sparse a dense por muestra si la matriz no se densifica."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        row = self.X[idx]
        if sparse.issparse(row):
            x = np.asarray(row.todense(), dtype=np.float32).reshape(-1)
        else:
            x = np.asarray(row, dtype=np.float32).reshape(-1)
        y = int(self.y[idx])
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class TextClassifier(nn.Module):
    """Clasificador denso con dos capas ocultas, BatchNorm1d y Dropout para texto TF-IDF."""

    def __init__(self, input_dim, hidden1=1024, hidden2=256, n_classes=3, dropout=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def evaluate_model(model, loader, criterion, device):
    """Evaluar el modelo sobre un loader y devolver pérdida y métricas."""
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)
            preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            targets.append(yb.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    loss = total_loss / len(loader.dataset)
    acc = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, average='macro', zero_division=0)
    recall = recall_score(targets, preds, average='macro', zero_division=0)
    f1_macro = f1_score(targets, preds, average='macro', zero_division=0)
    f1_micro = f1_score(targets, preds, average='micro', zero_division=0)
    return loss, acc, precision, recall, f1_macro, f1_micro, preds, targets


def save_training_history(history):
    """Guardar el historial de entrenamiento en CSV."""
    pd.DataFrame(history).to_csv(TRAINING_HISTORY_PATH, index=False)


def save_config(input_dim, n_classes):
    """Guardar la configuración de entrenamiento para reproducibilidad."""
    config_lines = [
        f'seed={RANDOM_STATE}',
        f'batch_size={BATCH_SIZE}',
        f'lr={LEARNING_RATE}',
        f'epochs={EPOCHS}',
        f'patience={PATIENCE}',
        f'architecture=TextClassifier(input_dim={input_dim}, hidden1=1024, hidden2=256, n_classes={n_classes}, dropout=0.5)',
        f'loss=CrossEntropyLoss(weight=class_weights)',
        f'optimizer=Adam',
        f'device={device}',
        'strategy=TF-IDF sparse matrix loaded from artifacts/X_tfidf.npz and converted to dense per batch if needed',
    ]
    TRAINING_CONFIG_PATH.write_text('\n'.join(config_lines), encoding='utf-8')


def plot_confusion_matrix(y_true, y_pred):
    """Guardar la matriz de confusión como imagen PNG."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['negative', 'positive'], rotation=45)
    plt.yticks(tick_marks, ['negative', 'positive'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='black')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()


def save_test_results(metrics, labels, preds):
    """Guardar resultados de test y reporte de clasificación."""
    with CLASSIFICATION_REPORT_PATH.open('w', encoding='utf-8') as f:
        f.write(classification_report(labels, preds, target_names=['negative', 'positive']))
    with TEST_RESULTS_PATH.open('w', encoding='utf-8') as f:
        f.write(f'accuracy={metrics["accuracy"]:.4f}\n')
        f.write(f'precision_macro={metrics["precision"]:.4f}\n')
        f.write(f'recall_macro={metrics["recall"]:.4f}\n')
        f.write(f'f1_macro={metrics["f1_macro"]:.4f}\n')
        f.write(f'f1_micro={metrics["f1_micro"]:.4f}\n')


def build_misclassified_examples(ids, texts, true_labels, pred_labels, pred_probs):
    """Guardar los ejemplos mal clasificados con probabilidad de predicción."""
    df = pd.DataFrame({
        'id': ids,
        'text_lemma': texts,
        'true_label': true_labels,
        'pred_label': pred_labels,
        'pred_prob': pred_probs,
    })
    df = df[df['true_label'] != df['pred_label']].sort_values('pred_prob', ascending=False).head(20)
    df.to_csv(MISCLASSIFIED_PATH, index=False, encoding='utf-8')
    return df


def main():
    set_seed(RANDOM_STATE)
    X_train, y_train, X_val, y_val, X_test, y_test = load_or_prepare_split()

    # Si la matriz es muy grande, se conserva sparse y se convierte por batch; si no, se densifica.
    if sparse.issparse(X_train):
        X_train_dense = to_dense_if_possible(X_train)
        X_val_dense = to_dense_if_possible(X_val)
        X_test_dense = to_dense_if_possible(X_test)
    else:
        X_train_dense = X_train.astype(np.float32)
        X_val_dense = X_val.astype(np.float32)
        X_test_dense = X_test.astype(np.float32)

    # Preparar datasets y data loaders.
    train_dataset = SparseRowDataset(X_train_dense, y_train)
    val_dataset = SparseRowDataset(X_val_dense, y_val)
    test_dataset = SparseRowDataset(X_test_dense, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    input_dim = X_train_dense.shape[1]
    n_classes = int(np.unique(y_train).max()) + 1
    global INPUT_DIM, N_CLASSES
    INPUT_DIM = input_dim
    N_CLASSES = n_classes

    model = TextClassifier(input_dim=input_dim, hidden1=1024, hidden2=256, n_classes=n_classes).to(device)
    class_weights = np.load(CLASS_WEIGHTS_PATH)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    save_config(input_dim, n_classes)

    history = []
    best_val_loss = float('inf')
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == yb).sum().item()
            train_total += yb.size(0)

        train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        val_loss, val_acc, val_precision, val_recall, val_f1_macro, val_f1_micro, _, _ = evaluate_model(model, val_loader, criterion, device)
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'val_f1_macro': val_f1_macro,
            'lr': LEARNING_RATE,
        })
        save_training_history(history)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(best_state, BEST_MODEL_PATH)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f'Early stopping triggered at epoch {epoch + 1}')
                break

    # Cargar mejor checkpoint y evaluar.
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    test_loss, test_acc, test_precision, test_recall, test_f1_macro, test_f1_micro, test_preds, test_targets = evaluate_model(model, test_loader, criterion, device)

    save_test_results({
        'accuracy': test_acc,
        'precision': test_precision,
        'recall': test_recall,
        'f1_macro': test_f1_macro,
        'f1_micro': test_f1_micro,
    }, test_targets, test_preds)
    plot_confusion_matrix(test_targets, test_preds)

    # Construir ejemplos mal clasificados usando el test set original.
    test_ids = np.arange(len(y_test))
    test_texts = [str(x) for x in np.array([None] * len(y_test))]
    # Si existe el CSV procesado, se intenta usar el texto lematizado del mismo índice.
    label_path = PROCESSED_DIR / 'sample_5000_labeled.csv'
    if label_path.exists():
        labeled = pd.read_csv(label_path)
        if 'text_lemma' in labeled.columns and len(labeled) >= len(y_test):
            test_texts = [str(text) for text in labeled.iloc[:len(y_test)]['text_lemma'].tolist()]
            test_ids = labeled.iloc[:len(y_test)]['id'].tolist()

    probs = torch.softmax(model(torch.tensor(X_test_dense.astype(np.float32), device=device)), dim=1).cpu().detach().numpy()
    pred_probs = probs[np.arange(len(test_preds)), test_preds]
    build_misclassified_examples(test_ids, test_texts, test_targets, test_preds, pred_probs)

    print(f'Best epoch: {best_epoch} (val_loss: {best_val_loss:.4f})')
    print(f'Test accuracy: {test_acc:.4f}')
    print(f'Test precision macro: {test_precision:.4f}')
    print(f'Test recall macro: {test_recall:.4f}')
    print(f'Test f1 macro: {test_f1_macro:.4f}')
    print(f'Artifacts saved: {BEST_MODEL_PATH}, {TRAINING_HISTORY_PATH}, {TRAINING_CONFIG_PATH}, {CLASSIFICATION_REPORT_PATH}, {CONFUSION_MATRIX_PATH}, {TEST_RESULTS_PATH}, {MISCLASSIFIED_PATH}')
    print('Training completed successfully.')


if __name__ == '__main__':
    main()
