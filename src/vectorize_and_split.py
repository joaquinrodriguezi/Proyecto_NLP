# Bloque de uso: ejecutar desde la raíz del proyecto con:
# python src/step4_vectorize_and_split.py
# Dependencias: pandas, numpy, joblib, scikit-learn, nltk, scipy, torch (opcional para .pt)
# Instalar paquetes: python -m pip install pandas numpy joblib scikit-learn nltk scipy
# Si falta torch y desea .pt: python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

import datetime
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Intento de importar torch para generar archivos .pt.
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Forzar salida UTF-8 en Windows para evitar errores de consola con caracteres Unicode.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Rutas de entrada y salida.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = PROCESSED_DIR / 'sample_5000_lemmatized.csv'
LABELED_FILE = PROCESSED_DIR / 'sample_5000_labeled.csv'
LABELING_SAMPLE_FILE = ARTIFACTS_DIR / 'labeling_sample_300.csv'
VECTORIZER_FILE = ARTIFACTS_DIR / 'tfidf_vectorizer.pkl'
X_TFIDF_FILE = ARTIFACTS_DIR / 'X_tfidf.npz'
Y_FILE = ARTIFACTS_DIR / 'y.npy'
CLASS_WEIGHTS_FILE = ARTIFACTS_DIR / 'class_weights.npy'
FEATURE_INFO_FILE = ARTIFACTS_DIR / 'feature_info.txt'
TOP_FEATURES_GLOBAL_FILE = ARTIFACTS_DIR / 'top_features_global.csv'
TOP_FEATURES_BY_CLASS_FILE = ARTIFACTS_DIR / 'top_features_by_class.csv'
SPLIT_INFO_FILE = ARTIFACTS_DIR / 'train_val_test_split_info.txt'
VECTOR_SUMMARY_FILE = ARTIFACTS_DIR / 'vectorization_summary.txt'
TRAIN_PT = PROCESSED_DIR / 'train.pt'
VAL_PT = PROCESSED_DIR / 'val.pt'
TEST_PT = PROCESSED_DIR / 'test.pt'

# Parámetros semilla y TF-IDF.
RANDOM_STATE = 42
TFIDF_PARAMS = {
    'max_features': 20000,
    'ngram_range': (1, 2),
    'min_df': 5,
    'sublinear_tf': True,
    'norm': 'l2',
    'stop_words': None,
}

# Estos umbrales se usan para etiquetas de VADER.
POSITIVE_THRESHOLD = 0.05
NEGATIVE_THRESHOLD = -0.05


def ensure_nltk_resources():
    """Descargar recursos necesarios de NLTK si faltan."""
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        nltk.download('vader_lexicon')


def load_data(path: Path) -> pd.DataFrame:
    """Cargar archivo lematizado y comprobar columnas esperadas."""
    df = pd.read_csv(path, encoding='utf-8')
    expected = ['id', 'text_clean', 'text_emojis_mapped', 'text_lemma', 'n_tokens']
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f'Faltan columnas esperadas en {path}: {missing}')
    return df


def vadar_label(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple[float, str]:
    """Calcular score compound y asignar etiqueta VADER."""
    if pd.isna(text):
        text = ''
    scores = analyzer.polarity_scores(str(text))
    compound = float(scores['compound'])
    if compound >= POSITIVE_THRESHOLD:
        label = 'positive'
    elif compound <= NEGATIVE_THRESHOLD:
        label = 'negative'
    else:
        label = 'neutral'
    return compound, label


def sample_labeling(df: pd.DataFrame) -> pd.DataFrame:
    """Guardar una muestra aleatoria para revisión manual."""
    sample = df.sample(n=300, random_state=RANDOM_STATE).copy()
    sample = sample[['id', 'text_lemma', 'compound', 'label']]
    sample.to_csv(LABELING_SAMPLE_FILE, index=False, encoding='utf-8')
    return sample


def get_label_counts(df: pd.DataFrame) -> pd.Series:
    """Contar cada etiqueta y devolver proporciones."""
    counts = df['label'].value_counts().sort_index()
    return counts


def build_vectorizer(stopwords_path: Path | None = None) -> TfidfVectorizer:
    """Crear vectorizador TF-IDF con parámetros fijos."""
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    if stopwords_path and stopwords_path.exists():
        stopwords = [w.strip() for w in stopwords_path.read_text(encoding='utf-8').splitlines() if w.strip()]
        if stopwords:
            vectorizer.set_params(stop_words=stopwords)
    return vectorizer


def top_features_global(vectorizer: TfidfVectorizer, X: sparse.spmatrix, top_n: int = 50) -> pd.DataFrame:
    """Calcular las mejores features globales por promedio TF-IDF."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    avg_tfidf = np.asarray(X.mean(axis=0)).ravel()
    top_idx = np.argsort(avg_tfidf)[::-1][:top_n]
    return pd.DataFrame({
        'feature': feature_names[top_idx],
        'avg_tfidf': avg_tfidf[top_idx],
    })


def top_features_by_label(vectorizer: TfidfVectorizer, X: sparse.spmatrix, labels: pd.Series, top_n: int = 20) -> pd.DataFrame:
    """Calcular las mejores features medias por clase original."""
    rows = []
    feature_names = np.array(vectorizer.get_feature_names_out())
    for label in sorted(labels.unique()):
        mask = labels == label
        if mask.sum() == 0:
            continue
        idx = np.flatnonzero(mask.to_numpy())
        X_label = X[idx]
        avg_tfidf = np.asarray(X_label.mean(axis=0)).ravel()
        top_idx = np.argsort(avg_tfidf)[::-1][:top_n]
        for rank, feature_idx in enumerate(top_idx, start=1):
            rows.append({
                'label': label,
                'rank': rank,
                'feature': feature_names[feature_idx],
                'avg_tfidf': avg_tfidf[feature_idx],
            })
    return pd.DataFrame(rows)


def sparse_to_torch(coo: sparse.coo_matrix) -> 'torch.Tensor':
    """Convertir matriz scipy COO a tensor torch sparse COO."""
    values = torch.tensor(coo.data.astype(np.float32))
    indices = torch.tensor(np.vstack((coo.row, coo.col)), dtype=torch.int64)
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def save_split_torch(path: Path, X: sparse.spmatrix, y: np.ndarray) -> None:
    """Guardar split como .pt usando sparse tensor si torch está disponible."""
    if not TORCH_AVAILABLE:
        return
    coo = X.tocoo()
    sparse_tensor = sparse_to_torch(coo)
    y_tensor = torch.tensor(y, dtype=torch.long)
    torch.save({'X': sparse_tensor, 'y': y_tensor}, path)


def main():
    ensure_nltk_resources()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f'No se encuentra el archivo de entrada {INPUT_FILE}')

    df = load_data(INPUT_FILE)
    print(f'Data loaded: {INPUT_FILE} (shape: {df.shape})')

    analyzer = SentimentIntensityAnalyzer()
    results = df['text_lemma'].astype(str).apply(lambda text: vadar_label(text, analyzer))
    df[['compound', 'label']] = pd.DataFrame(results.tolist(), index=df.index)

    df[['id', 'text_clean', 'text_lemma', 'compound', 'label']].to_csv(
        LABELED_FILE, index=False, encoding='utf-8'
    )
    print(f'Labeled data saved: {LABELED_FILE} (shape: {df.shape})')

    counts = get_label_counts(df)
    total = counts.sum()
    for label, count in counts.items():
        print(f'  {label}: {count} ({count / total:.2%})')

    sample = sample_labeling(df)
    print(f'Labeling sample saved: {LABELING_SAMPLE_FILE} (300 rows)')

    print('\nEjemplos de etiquetas:')
    for _, row in sample.groupby('label', group_keys=False).head(4).iterrows():
        print(f"id={row['id']} label={row['label']} compound={row['compound']:.3f} text_lemma={row['text_lemma'][:80]!r}")

    vectorizer = build_vectorizer(PROJECT_ROOT / 'artifacts' / 'stopwords_final.txt')
    X_tfidf = vectorizer.fit_transform(df['text_lemma'].fillna(''))
    print(f'TF-IDF matrix shape: {X_tfidf.shape}')
    joblib.dump(vectorizer, VECTORIZER_FILE)
    print(f'TF-IDF vectorizer saved: {VECTORIZER_FILE} (n_features: {X_tfidf.shape[1]})')

    sparse.save_npz(X_TFIDF_FILE, X_tfidf)
    np.save(Y_FILE, np.where(df['label'] == 'negative', 0, 1).astype(np.int64))
    print(f'X_tfidf saved: {X_TFIDF_FILE}')
    print(f'y.npy saved: {Y_FILE}')

    # División estratificada usando etiquetas binarias performance-friendly.
    y_binary = np.where(df['label'] == 'negative', 0, 1)
    X_train, X_temp, y_train, y_temp, idx_train, idx_temp = train_test_split(
        X_tfidf, y_binary, df.index.values, test_size=0.15, stratify=y_binary, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
        X_temp, y_temp, idx_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE
    )

    # Guardar particiones para PyTorch; si torch no está instalado, escribir .npz y documentar.
    if TORCH_AVAILABLE:
        save_split_torch(TRAIN_PT, X_train, y_train)
        save_split_torch(VAL_PT, X_val, y_val)
        save_split_torch(TEST_PT, X_test, y_test)
        print(f'Train/Val/Test saved: {TRAIN_PT}, {VAL_PT}, {TEST_PT}')
    else:
        print('torch no disponible: no se generaron archivos .pt; ver train_val_test_split_info.txt para detalles.')

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    np.save(CLASS_WEIGHTS_FILE, class_weights)
    print(f'Class weights saved: {CLASS_WEIGHTS_FILE}')
    for cls, weight in zip(np.unique(y_train), class_weights):
        print(f'  class {cls}: weight {weight:.4f}')
    print('Se recomienda usar nn.CrossEntropyLoss(weight=class_weights) para mitigar desbalance.')

    feature_global = top_features_global(vectorizer, X_tfidf, top_n=50)
    feature_global.to_csv(TOP_FEATURES_GLOBAL_FILE, index=False, encoding='utf-8')
    feature_global.to_csv(FEATURE_INFO_FILE, index=False, encoding='utf-8')

    feature_by_class = top_features_by_label(vectorizer, X_tfidf, df['label'], top_n=20)
    feature_by_class.to_csv(TOP_FEATURES_BY_CLASS_FILE, index=False, encoding='utf-8')

    with FEATURE_INFO_FILE.open('w', encoding='utf-8') as f:
        f.write(f'Número de features TF-IDF: {X_tfidf.shape[1]}\n')
        f.write('Top 50 features globales:\n')
        feature_global.to_string(f, index=False)

    with SPLIT_INFO_FILE.open('w', encoding='utf-8') as f:
        f.write('Train/Val/Test split counts\n')
        f.write(f'train: {X_train.shape[0]}\n')
        f.write(f'val: {X_val.shape[0]}\n')
        f.write(f'test: {X_test.shape[0]}\n')
        f.write('\nDistribution binary labels by split\n')
        for name, y in [('train', y_train), ('val', y_val), ('test', y_test)]:
            unique, split_counts = np.unique(y, return_counts=True)
            f.write(f'{name}: ' + ', '.join(f'{int(c)} (class {int(u)})' for u, c in zip(unique, split_counts)) + '\n')
        f.write('\nTorch available: ' + str(TORCH_AVAILABLE) + '\n')

    summary_lines = [
        f'Fecha: {datetime.datetime.utcnow().isoformat()}Z',
        'Vectorization summary',
        '---------------------',
        f'Parameters: {TFIDF_PARAMS}',
        'Decision: se usa TF-IDF vs conteo porque captura importancia de términos normalizada y permite n-grams.',
        'Decision: el neutral se agrupa con positive para clasificación binaria; esto reduce la clase negativa a 0 y positive/neutral a 1.',
        f'Neutral -> positive: {int((df["label"] == "neutral").sum())} ejemplos.',
        f'torch disponible: {TORCH_AVAILABLE}',
    ]
    VECTOR_SUMMARY_FILE.write_text('\n'.join(summary_lines), encoding='utf-8')

    print(f'Feature info saved: {FEATURE_INFO_FILE}')
    print(f'Top global features saved: {TOP_FEATURES_GLOBAL_FILE}')
    print(f'Top features by class saved: {TOP_FEATURES_BY_CLASS_FILE}')
    print(f'Split info saved: {SPLIT_INFO_FILE}')
    print(f'Vectorization summary saved: {VECTOR_SUMMARY_FILE}')

    labels_by_split = {
        'train': pd.Series(y_train).value_counts().sort_index(),
        'val': pd.Series(y_val).value_counts().sort_index(),
        'test': pd.Series(y_test).value_counts().sort_index(),
    }
    print('\nResumen final:')
    print(f'  Rutas generadas: {LABELED_FILE}, {LABELING_SAMPLE_FILE}, {VECTORIZER_FILE}, {X_TFIDF_FILE}, {Y_FILE}, {SPLIT_INFO_FILE}, {VECTOR_SUMMARY_FILE}, {FEATURE_INFO_FILE}, {TOP_FEATURES_GLOBAL_FILE}, {TOP_FEATURES_BY_CLASS_FILE}, {CLASS_WEIGHTS_FILE}')
    if TORCH_AVAILABLE:
        print(f'  PyTorch splits: {TRAIN_PT}, {VAL_PT}, {TEST_PT}')
    print('  Label distribution:')
    for label, count in counts.items():
        print(f'    {label}: {count} ({count / total:.2%})')
    print('  Split sizes: train=%d, val=%d, test=%d' % (X_train.shape[0], X_val.shape[0], X_test.shape[0]))
    print(f'  TF-IDF features: {X_tfidf.shape[1]}')
    print('  Recomendación: usar class_weights en nn.CrossEntropyLoss(weight=...)')


if __name__ == '__main__':
    main()
