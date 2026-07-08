# Bloque de uso: ejecutar desde la raíz del proyecto con:
# python src/analysis_plots.py
# Dependencias: pandas, numpy, matplotlib, seaborn, sklearn, joblib, torch, scipy
# Este script lee los artefactos de entrenamiento y genera gráficos y resúmenes para el informe final.

import os
import sys
import json
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import sparse
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import joblib

# Rutas base del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC_DIR = PROJECT_ROOT / 'src'
if str(PROJECT_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC_DIR))

# Forzar salida UTF-8 en Windows para evitar problemas de consola.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Rutas de entrada y salida.
TRAINING_HISTORY_CSV = ARTIFACTS_DIR / 'training_history.csv'
TRAINING_HISTORY_JSON = ARTIFACTS_DIR / 'training_history.json'
TRAINING_HISTORY_PKL = ARTIFACTS_DIR / 'training_history.pkl'
CLASSIFICATION_REPORT_PATH = ARTIFACTS_DIR / 'classification_report.txt'
MISCLASSIFIED_PATH = ARTIFACTS_DIR / 'misclassified_examples.csv'
TEST_RESULTS_PATH = ARTIFACTS_DIR / 'test_results.txt'
MODEL_PATH = ARTIFACTS_DIR / 'best_model.pt'
VECTORIZER_PATH = ARTIFACTS_DIR / 'tfidf_vectorizer.pkl'
X_TFIDF_PATH = ARTIFACTS_DIR / 'X_tfidf.npz'
Y_PATH = ARTIFACTS_DIR / 'y.npy'
TEST_PT_PATH = PROCESSED_DIR / 'test.pt'

# Archivos generados.
LOSS_PLOT_PATH = ARTIFACTS_DIR / 'plot_loss_curves.png'
ACCURACY_PLOT_PATH = ARTIFACTS_DIR / 'plot_accuracy_curves.png'
F1_PLOT_PATH = ARTIFACTS_DIR / 'plot_f1_macro_curve.png'
CONFUSION_MATRIX_PATH = ARTIFACTS_DIR / 'confusion_matrix_annotated.png'
CONFUSION_VALUES_PATH = ARTIFACTS_DIR / 'confusion_matrix_values.csv'
CALIBRATION_PLOT_PATH = ARTIFACTS_DIR / 'calibration_plot.png'
CALIBRATION_SCORES_PATH = ARTIFACTS_DIR / 'calibration_scores.csv'
TOP_FEATURES_GLOBAL_PLOT_PATH = ARTIFACTS_DIR / 'top_features_global.png'
TOP_FEATURES_GLOBAL_CSV_PATH = ARTIFACTS_DIR / 'top_features_global.csv'
TOP_FEATURES_BY_CLASS_PLOT_PATH = ARTIFACTS_DIR / 'top_features_by_class.png'
TOP_FEATURES_BY_CLASS_CSV_PATH = ARTIFACTS_DIR / 'top_features_by_class.csv'
MISCLASSIFIED_CATEGORIZED_PATH = ARTIFACTS_DIR / 'misclassified_categorized.csv'
REPRESENTATIVE_ERRORS_PATH = ARTIFACTS_DIR / 'misclassified_representative_50.csv'
MISCLASSIFIED_GRID_PATH = ARTIFACTS_DIR / 'misclassified_examples_grid.png'
MISCLASSIFIED_DOC_PATH = ARTIFACTS_DIR / 'misclassified_examples_for_doc.csv'
ANALYSIS_SUMMARY_PATH = ARTIFACTS_DIR / 'analysis_summary.txt'

RANDOM_STATE = 42
LABELS = ['negative', 'positive']


def set_seed(seed: int = 42) -> None:
    """Fijar la semilla para reproducibilidad."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_history() -> pd.DataFrame:
    """Cargar el historial de entrenamiento desde CSV, JSON o PKL."""
    candidates = [TRAINING_HISTORY_CSV, TRAINING_HISTORY_JSON, TRAINING_HISTORY_PKL]
    for path in candidates:
        if path.exists():
            if path.suffix == '.csv':
                return pd.read_csv(path)
            if path.suffix == '.json':
                with path.open('r', encoding='utf-8') as handle:
                    data = json.load(handle)
                return pd.DataFrame(data)
            if path.suffix == '.pkl':
                with path.open('rb') as handle:
                    data = pickle.load(handle)
                return pd.DataFrame(data)
    raise FileNotFoundError('No se encontró training_history en artifacts/.')


def load_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Cargar los datos de test desde data/processed/test.pt o desde los artefactos TF-IDF."""
    if TEST_PT_PATH.exists():
        payload = torch.load(TEST_PT_PATH, map_location='cpu')
        X = payload['X']
        y = payload['y']
        if isinstance(X, torch.Tensor):
            if X.layout == torch.sparse_coo:
                X = X.to_dense()
            X = X.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        return np.asarray(X), np.asarray(y)

    if X_TFIDF_PATH.exists() and Y_PATH.exists() and MODEL_PATH.exists():
        X = sparse.load_npz(X_TFIDF_PATH)
        y = np.load(Y_PATH)
        return X, y
    raise FileNotFoundError('No se encontraron los datos de test para recalcular predicciones.')


def load_model_and_vectorizer() -> tuple[torch.nn.Module, object]:
    """Cargar el mejor modelo y el vectorizer TF-IDF."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError('No se encontró artifacts/best_model.pt')
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError('No se encontró artifacts/tfidf_vectorizer.pkl')

    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, dict) and any(k.startswith('fc') or k.startswith('bn') for k in checkpoint.keys()):
        state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Recrear la topología del modelo usado durante el entrenamiento con la dimensión real de TF-IDF.
    from train import TextClassifier
    X_full = sparse.load_npz(X_TFIDF_PATH)
    input_dim = X_full.shape[1]
    model = TextClassifier(input_dim=input_dim, hidden1=1024, hidden2=256, n_classes=2)
    model.load_state_dict(state_dict)
    model.eval()
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def save_with_metadata(df: pd.DataFrame, path: Path) -> None:
    """Guardar un DataFrame con columna id única y timestamp de creación."""
    if 'id' not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0, 'id', [f'{path.stem}_{i:05d}' for i in range(len(df))])
    else:
        df = df.copy()
        df['id'] = [f'{path.stem}_{i:05d}' for i in range(len(df))]
    df['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df.to_csv(path, index=False, encoding='utf-8')


def plot_training_curves(history: pd.DataFrame) -> None:
    """Guardar las curvas de pérdida, precisión y F1 macro para el informe."""
    history = history.copy()
    best_idx = history['val_loss'].idxmin()
    best_epoch = int(history.loc[best_idx, 'epoch'])
    best_val_loss = float(history.loc[best_idx, 'val_loss'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
    axes[0].plot(history['epoch'], history['train_loss'], marker='o', label='train_loss')
    axes[0].plot(history['epoch'], history['val_loss'], marker='o', label='val_loss')
    axes[0].set_title('Pérdida por época')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].axvline(best_epoch, color='gray', linestyle='--', alpha=0.7)
    axes[0].annotate(f'Best val_loss@{best_epoch}', xy=(best_epoch, best_val_loss), xytext=(best_epoch + 0.5, best_val_loss + 0.05), arrowprops=dict(arrowstyle='->'))

    axes[1].plot(history['epoch'], history['train_acc'], marker='o', label='train_acc')
    axes[1].plot(history['epoch'], history['val_acc'], marker='o', label='val_acc')
    axes[1].set_title('Accuracy por época')
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].axvline(best_epoch, color='gray', linestyle='--', alpha=0.7)

    axes[2].plot(history['epoch'], history['val_f1_macro'], marker='o', label='val_f1_macro')
    axes[2].set_title('F1 macro por época')
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('F1 macro')
    axes[2].legend()
    axes[2].axvline(best_epoch, color='gray', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close(fig)

    # Guardar curvas individuales para mayor claridad.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['epoch'], history['train_loss'], marker='o', label='train_loss')
    ax.plot(history['epoch'], history['val_loss'], marker='o', label='val_loss')
    ax.set_title('Curva de pérdida')
    ax.set_xlabel('Época')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.axvline(best_epoch, color='gray', linestyle='--', alpha=0.7)
    ax.annotate(f'Best val_loss@{best_epoch}', xy=(best_epoch, best_val_loss), xytext=(best_epoch + 0.5, best_val_loss + 0.05), arrowprops=dict(arrowstyle='->'))
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['epoch'], history['train_acc'], marker='o', label='train_acc')
    ax.plot(history['epoch'], history['val_acc'], marker='o', label='val_acc')
    ax.set_title('Curva de accuracy')
    ax.set_xlabel('Época')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.axvline(best_epoch, color='gray', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history['epoch'], history['val_f1_macro'], marker='o', label='val_f1_macro')
    ax.set_title('Curva de F1 macro')
    ax.set_xlabel('Época')
    ax.set_ylabel('F1 macro')
    ax.legend()
    ax.axvline(best_epoch, color='gray', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(F1_PLOT_PATH, dpi=150)
    plt.close(fig)

    return best_epoch, best_val_loss


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Generar matriz de confusión anotada y guardar valores numéricos."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    cm_df.index.name = 'true_label'
    cm_df.columns.name = 'pred_label'
    save_with_metadata(cm_df.reset_index(), CONFUSION_VALUES_PATH)

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=0.5, ax=ax, cbar=True)
    ax.set_title('Matriz de confusión')
    ax.set_xlabel('Predicción')
    ax.set_ylabel('Etiqueta real')
    ax.set_xticklabels(LABELS, rotation=45)
    ax.set_yticklabels(LABELS, rotation=0)

    # Añadir porcentajes por fila.
    row_sums = cm.sum(axis=1, keepdims=True)
    percentages = (cm / row_sums * 100).round(1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j + 0.5, i + 0.5, f'{cm[i, j]}\n({percentages[i, j]}%)', ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)


def build_calibration_plot(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Generar un diagrama de confiabilidad y guardar métricas de calibración."""
    bins = np.linspace(0.0, 1.0, 11)
    bin_indices = np.digitize(y_prob, bins[1:-1], right=True)
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

    rows = []
    for idx in range(len(bins) - 1):
        mask = bin_indices == idx
        if not np.any(mask):
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        count = mask.sum()
        ece = abs(acc - conf) * (count / len(y_true))
        rows.append({'bin': idx, 'confidence': conf, 'accuracy': acc, 'count': count, 'ece_contrib': ece})

    cal_df = pd.DataFrame(rows)
    cal_df.to_csv(CALIBRATION_SCORES_PATH, index=False, encoding='utf-8')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
    ax.plot(cal_df['confidence'], cal_df['accuracy'], marker='o', linewidth=2, label='Observed')
    ax.set_title('Reliability diagram para la clase positiva')
    ax.set_xlabel('Confianza promedio')
    ax.set_ylabel('Precisión observada')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    plt.savefig(CALIBRATION_PLOT_PATH, dpi=150)
    plt.close(fig)


def build_feature_importance_plots(X: sparse.spmatrix, vectorizer) -> None:
    """Guardar gráficas de importancia global y por clase a partir del TF-IDF."""
    feature_names = vectorizer.get_feature_names_out()
    X_full = sparse.load_npz(X_TFIDF_PATH)
    y_full = np.load(Y_PATH)
    if sparse.issparse(X):
        X_mean = np.asarray(X_full.mean(axis=0)).ravel()
    else:
        X_mean = np.asarray(X_full.mean(axis=0)).ravel()
    global_df = pd.DataFrame({'feature': feature_names, 'mean_tfidf': X_mean})
    global_df = global_df.sort_values('mean_tfidf', ascending=False).head(30)
    save_with_metadata(global_df, TOP_FEATURES_GLOBAL_CSV_PATH)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=global_df, x='mean_tfidf', y='feature', palette='viridis', ax=ax)
    ax.set_title('Top 30 features TF-IDF globales')
    ax.set_xlabel('Promedio TF-IDF')
    ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.savefig(TOP_FEATURES_GLOBAL_PLOT_PATH, dpi=150)
    plt.close(fig)

    # Features por clase.
    class_dfs = []
    for cls in sorted(np.unique(y_full)):
        mask = y_full == cls
        if sparse.issparse(X):
            cls_mean = np.asarray(X_full[mask].mean(axis=0)).ravel()
        else:
            cls_mean = np.asarray(X_full[mask].mean(axis=0)).ravel()
        cls_df = pd.DataFrame({'class': int(cls), 'feature': feature_names, 'mean_tfidf': cls_mean})
        cls_df = cls_df.sort_values('mean_tfidf', ascending=False).head(20)
        cls_df['class_label'] = LABELS[int(cls)] if 0 <= int(cls) < len(LABELS) else int(cls)
        class_dfs.append(cls_df)

    class_df = pd.concat(class_dfs, ignore_index=True)
    save_with_metadata(class_df, TOP_FEATURES_BY_CLASS_CSV_PATH)

    fig, axes = plt.subplots(1, len(np.unique(y_full)), figsize=(14, 4), sharey=True)
    for ax, cls in zip(axes, sorted(np.unique(y_full))):
        subset = class_df[class_df['class'] == cls].head(12)
        sns.barplot(data=subset, x='mean_tfidf', y='feature', palette='viridis', ax=ax)
        ax.set_title(f'Top 20 features clase {LABELS[int(cls)]}')
        ax.set_xlabel('Promedio TF-IDF')
        ax.set_ylabel('Feature')
    plt.tight_layout()
    plt.savefig(TOP_FEATURES_BY_CLASS_PLOT_PATH, dpi=150)
    plt.close(fig)


def categorize_error(text: str, true_label: int, pred_label: int, pred_prob: float) -> tuple[str, str]:
    """Aplicar heurísticas simples para categorizar errores de clasificación."""
    text = str(text).lower()
    notes = ''

    positive_emojis = ['joy', 'smile', 'laugh', 'love', 'heart', 'party', 'duck', 'thumbs', 'grin']
    negative_emojis = ['angry', 'frown', 'cry', 'dead', 'sweat', 'upset', 'sick', 'vomit']
    sarcasm_markers = ['yeah right', 'sure', 'as if', 'right', 'lol', 'oh sure']

    has_emoji = 'emoji_' in text or 'emoji' in text
    text_tokens = [token for token in text.replace('-', ' ').split() if token]
    has_negation = any(token in text for token in ['not', 'no', 'never', 'dont', 'doesnt', 'ain\'t', 'isn\'t'])
    has_intensifier = any(token in text for token in ['very', 'so', 'really', 'super', 'totally', 'extremely'])

    if has_emoji:
        emoji_polarity = 1 if any(marker in text for marker in positive_emojis) else 0
        if any(marker in text for marker in negative_emojis):
            emoji_polarity = 0
        if emoji_polarity != true_label:
            return 'emoji_conflict', f'emoji polarity={emoji_polarity} vs true_label={true_label}'

    if any(marker in text for marker in sarcasm_markers):
        return 'sarcasm', 'palabras clave de sarcasmo detectadas'

    if pred_prob < 0.6 and len(text_tokens) <= 3:
        return 'label_noise', 'probabilidad baja y texto muy corto'

    if len(text_tokens) <= 3 or len(text_tokens) <= 4 and not any(token in text for token in ['good', 'bad', 'love', 'hate', 'win', 'lose', 'fight', 'great', 'terrible']):
        return 'ambiguous', 'texto corto o poco contexto'

    if has_negation and has_intensifier:
        return 'sarcasm', 'contraste entre negación e intensificador'

    return 'other', notes


def build_error_analysis(misclassified: pd.DataFrame) -> None:
    """Categorizar errores y guardar ejemplos representativos."""
    misclassified = misclassified.copy()
    misclassified['text_for_categorization'] = misclassified['text_lemma'].fillna('')
    misclassified['category'] = ''
    misclassified['notes'] = ''

    for idx, row in misclassified.iterrows():
        category, notes = categorize_error(row['text_for_categorization'], int(row['true_label']), int(row['pred_label']), float(row['pred_prob']))
        misclassified.at[idx, 'category'] = category
        misclassified.at[idx, 'notes'] = notes

    save_with_metadata(misclassified[['id', 'text_lemma', 'true_label', 'pred_label', 'pred_prob', 'category', 'notes']], MISCLASSIFIED_CATEGORIZED_PATH)

    rep_rows = []
    categories = sorted(misclassified['category'].dropna().unique().tolist())
    if not categories:
        categories = ['other']
    for category in categories:
        subset = misclassified[misclassified['category'] == category]
        if subset.empty:
            continue
        rep_rows.append(subset.sample(n=min(8, len(subset)), random_state=RANDOM_STATE))
    representative = pd.concat(rep_rows, ignore_index=True) if rep_rows else misclassified.head(50)
    representative = representative.head(50)
    save_with_metadata(representative[['id', 'text_lemma', 'true_label', 'pred_label', 'pred_prob', 'category', 'notes']], REPRESENTATIVE_ERRORS_PATH)

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    sample = representative.head(12).copy()
    for ax, row in zip(axes, sample.itertuples(index=False)):
        text = str(getattr(row, 'text_lemma', '') or '')
        text = (text[:120] + '...') if len(text) > 120 else text
        ax.text(0.02, 0.8, f'text: {text}', ha='left', va='top', wrap=True)
        ax.text(0.02, 0.55, f'true={getattr(row, "true_label")}', ha='left', va='top')
        ax.text(0.02, 0.35, f'pred={getattr(row, "pred_label")}', ha='left', va='top')
        ax.text(0.02, 0.15, f'prob={getattr(row, "pred_prob"):.3f}', ha='left', va='top')
        ax.text(0.02, 0.0, f'cat={getattr(row, "category")}', ha='left', va='top')
        ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(MISCLASSIFIED_GRID_PATH, dpi=150)
    plt.close(fig)

    doc_df = representative[['text_lemma', 'true_label', 'pred_label', 'pred_prob', 'category', 'notes']].copy()
    doc_df.columns = ['text', 'true_label', 'pred_label', 'pred_prob', 'category', 'notes']
    save_with_metadata(doc_df.head(20), MISCLASSIFIED_DOC_PATH)


def build_summary(history: pd.DataFrame, misclassified: pd.DataFrame, test_metrics: dict, recalculated: bool) -> None:
    """Crear un resumen listo para pegar en el informe final."""
    category_counts = misclassified['category'].value_counts().sort_index()
    lines = []
    lines.append('Resumen de análisis post-entrenamiento')
    lines.append('====================================')
    lines.append(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('')
    lines.append(f'Mejor época: {int(history["epoch"].iloc[history["val_loss"].idxmin()])}')
    lines.append(f'Val loss mínima: {history["val_loss"].min():.4f}')
    lines.append(f'Accuracy test: {test_metrics.get("accuracy", "n/a"):.4f}')
    lines.append(f'Precision macro test: {test_metrics.get("precision", "n/a"):.4f}')
    lines.append(f'Recall macro test: {test_metrics.get("recall", "n/a"):.4f}')
    lines.append(f'F1 macro test: {test_metrics.get("f1_macro", "n/a"):.4f}')
    lines.append(f'Total de errores: {len(misclassified)}')
    lines.append('Distribución por categoría:')
    for cat, count in category_counts.items():
        lines.append(f'  - {cat}: {count}')
    lines.append('')
    if recalculated:
        lines.append('Predicciones recalculadas usando artifacts/best_model.pt + artifacts/X_tfidf.npz + data/processed/test.pt.')
    else:
        lines.append('Se usaron predicciones preexistentes del artefacto artifacts/confusion_matrix.png y artifacts/misclassified_examples.csv.')
    lines.append('')
    lines.append('Archivos generados:')
    generated_files = [
        LOSS_PLOT_PATH,
        ACCURACY_PLOT_PATH,
        F1_PLOT_PATH,
        CONFUSION_MATRIX_PATH,
        CONFUSION_VALUES_PATH,
        CALIBRATION_PLOT_PATH,
        CALIBRATION_SCORES_PATH,
        TOP_FEATURES_GLOBAL_PLOT_PATH,
        TOP_FEATURES_GLOBAL_CSV_PATH,
        TOP_FEATURES_BY_CLASS_PLOT_PATH,
        TOP_FEATURES_BY_CLASS_CSV_PATH,
        MISCLASSIFIED_CATEGORIZED_PATH,
        REPRESENTATIVE_ERRORS_PATH,
        MISCLASSIFIED_GRID_PATH,
        MISCLASSIFIED_DOC_PATH,
        ANALYSIS_SUMMARY_PATH,
    ]
    for path in generated_files:
        lines.append(f'  - {path.relative_to(PROJECT_ROOT)}')
    lines.append('')
    lines.append('Recomendaciones:')
    lines.append('  - Revisar etiquetas VADER en ejemplos ambiguos o con emojis conflictivos.')
    lines.append('  - Añadir 1k ejemplos manualmente anotados para reducir ruido de etiqueta.')
    lines.append('  - Probar TruncatedSVD o embeddings ligeros para mejorar separación semántica.')
    lines.append('  - Evaluar un modelo de clasificación por clases con balanceo adicional para errores de sarcasmo.')

    ANALYSIS_SUMMARY_PATH.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


def main() -> None:
    """Ejecutar el análisis completo y guardar todos los artefactos requeridos."""
    set_seed(RANDOM_STATE)

    print('Cargando historial de entrenamiento...')
    history = load_training_history()
    print(history.head().to_string(index=False))

    print('Cargando reporte de clasificación y ejemplos mal clasificados...')
    classification_report = CLASSIFICATION_REPORT_PATH.read_text(encoding='utf-8')
    print(classification_report)
    misclassified = pd.read_csv(MISCLASSIFIED_PATH)
    if 'text_lemma' not in misclassified.columns:
        misclassified['text_lemma'] = misclassified.get('text', '')

    print('Generando curvas de entrenamiento...')
    best_epoch, best_val_loss = plot_training_curves(history)
    print(f'Mejor época: {best_epoch} con val_loss {best_val_loss:.4f}')

    print('Recalculando predicciones sobre test...')
    X_test, y_test = load_test_data()
    model, vectorizer = load_model_and_vectorizer()
    if isinstance(X_test, sparse.spmatrix):
        X_test_dense = X_test.astype(np.float32).toarray()
    else:
        X_test_dense = np.asarray(X_test, dtype=np.float32)

    if isinstance(model, dict):
        model = model['model'] if 'model' in model else torch.nn.Module()
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test_dense, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    pred_probs = probs[:, 1]
    y_pred = (pred_probs >= 0.5).astype(int)

    print('Guardando matriz de confusión y calibración...')
    build_confusion_matrix(y_test, y_pred)
    build_calibration_plot(y_test, pred_probs)

    print('Guardando importancia de features...')
    build_feature_importance_plots(X_test, vectorizer)

    print('Categorizando errores...')
    build_error_analysis(misclassified)

    print('Creando resumen final...')
    test_metrics = {
        'accuracy': float((y_test == y_pred).mean()),
        'precision': float(precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)[0]),
        'recall': float(precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)[1]),
        'f1_macro': float(precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)[2]),
    }
    misclassified_categorized = pd.read_csv(MISCLASSIFIED_CATEGORIZED_PATH)
    build_summary(history, misclassified_categorized, test_metrics, recalculated=True)

    print('Archivos generados:')
    generated_paths = [
        LOSS_PLOT_PATH,
        ACCURACY_PLOT_PATH,
        F1_PLOT_PATH,
        CONFUSION_MATRIX_PATH,
        CONFUSION_VALUES_PATH,
        CALIBRATION_PLOT_PATH,
        CALIBRATION_SCORES_PATH,
        TOP_FEATURES_GLOBAL_PLOT_PATH,
        TOP_FEATURES_GLOBAL_CSV_PATH,
        TOP_FEATURES_BY_CLASS_PLOT_PATH,
        TOP_FEATURES_BY_CLASS_CSV_PATH,
        MISCLASSIFIED_CATEGORIZED_PATH,
        REPRESENTATIVE_ERRORS_PATH,
        MISCLASSIFIED_GRID_PATH,
        MISCLASSIFIED_DOC_PATH,
        ANALYSIS_SUMMARY_PATH,
    ]
    for path in generated_paths:
        print(f'  - {path.relative_to(PROJECT_ROOT)} -> {path.exists()}')


if __name__ == '__main__':
    main()
