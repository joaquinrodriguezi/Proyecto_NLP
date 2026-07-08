# Bloque de uso: ejecutar este script desde la raíz del proyecto con:
# python src/eda.py
# Este script crea una submuestra reproducible del corpus y genera artefactos de EDA en artifacts/.

# Importar las bibliotecas necesarias para lectura, visualización, procesamiento básico y generación de notebook.
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import nbformat as nbf

# Definir rutas del proyecto y archivos de entrada/salida para trabajar de forma robusta.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
INPUT_FILE = RAW_DIR / "UFC_champion_Jon_Jones_retires_comments.csv"
SAMPLE_FILE = RAW_DIR / "sample_5000.csv"

# Crear carpetas de salida si aún no existen para no sobrescribir contenido previo.
for directory in [RAW_DIR, ARTIFACTS_DIR, NOTEBOOKS_DIR, SRC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Cargar el CSV completo o, si existe, la muestra previa guardada en el paso 1.
if INPUT_FILE.exists():
    df_full = pd.read_csv(INPUT_FILE, encoding="utf-8")
    print(f"CSV cargado desde: {INPUT_FILE}")
else:
    raise FileNotFoundError(f"No se encontró el archivo base: {INPUT_FILE}")

# Crear una submuestra aleatoria reproducible de 5.000 filas con semilla fija.
df = df_full.sample(n=5000, random_state=42).reset_index(drop=True)
df.to_csv(SAMPLE_FILE, index=False, encoding="utf-8")
print(f"Submuestra guardada: {SAMPLE_FILE} (shape: {df.shape})")

# Confirmar la columna de texto a usar; se prioriza content, luego text/comment/body/message.
text_column = None
for candidate in ["content", "text", "comment", "body", "message"]:
    if candidate in df.columns:
        text_column = candidate
        break
if text_column is None:
    for column_name in df.columns:
        if pd.api.types.is_object_dtype(df[column_name]):
            text_column = column_name
            break

# Confirmar si existe una columna de etiqueta entre los nombres típicos.
label_column = None
for candidate in ["label", "sentiment", "class", "category"]:
    if candidate in df.columns:
        label_column = candidate
        break

print(f"Columnas detectadas: {list(df.columns)}")
print(f"Columna de texto: {text_column}")
if label_column is None:
    print("Columna de etiqueta: sin etiqueta")
else:
    print(f"Columna de etiqueta: {label_column}")

# Preparar una columna de texto limpia para el análisis, preservando emojis.
text_series = df[text_column].fillna("").astype(str)

# Crear una distribución de clases simple para el caso sin etiqueta y para el caso supervisado.
if label_column is not None and df[label_column].nunique(dropna=False) >= 2:
    class_counts = df[label_column].fillna("NaN").value_counts(dropna=False)
    class_df = pd.DataFrame({"label": class_counts.index, "count": class_counts.values})
    class_df["pct"] = (class_df["count"] / len(df) * 100).round(2)
else:
    class_df = pd.DataFrame({"label": ["sin_etiqueta"], "count": [len(df)]})
    class_df["pct"] = [100.0]
class_df.to_csv(ARTIFACTS_DIR / "class_distribution.csv", index=False)

plt.figure(figsize=(8, 5))
sns.barplot(data=class_df, x="label", y="count")
plt.title("Distribución de clases")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "class_distribution.png", dpi=200)
plt.close()
print("Class distribution saved: artifacts/class_distribution.png, artifacts/class_distribution.csv")

# Calcular métricas de longitud en caracteres y palabras.
df["n_chars"] = text_series.str.len()
df["n_words"] = text_series.str.split().str.len()

# Guardar estadísticas descriptivas de longitud en CSV.
length_stats = df[["n_chars", "n_words"]].describe(percentiles=[0.25, 0.5, 0.75]).T
length_stats.to_csv(ARTIFACTS_DIR / "length_stats.csv")
print("Length stats saved: artifacts/length_stats.csv")

# Guardar histograma de palabras y boxplot de caracteres por clase o general.
plt.figure(figsize=(8, 5))
sns.histplot(df["n_words"], bins=30, kde=True)
plt.title("Distribución de longitud en palabras")
plt.xlabel("Número de palabras")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "length_hist_words.png", dpi=200)
plt.close()

if label_column is not None and df[label_column].nunique(dropna=False) >= 2:
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=label_column, y="n_chars")
    plt.title("Distribución de longitud en caracteres por clase")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "length_boxplot_chars_by_class.png", dpi=200)
    plt.close()
else:
    plt.figure(figsize=(8, 5))
    sns.boxplot(y=df["n_chars"])
    plt.title("Distribución general de longitud en caracteres")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "length_boxplot_chars_by_class.png", dpi=200)
    plt.close()

# Contar registros con patrones problemáticos, incluyendo emojis y textos vacíos o cortos.
EMOJI_REGEX = r"[\U0001F300-\U0001FAFF\u2600-\u27BF]"
url_pattern = r"https?://|www\."
mention_pattern = r"(?<!\w)@\w+"
hashtag_pattern = r"(?<!\w)#\w+"
url_mask = text_series.str.contains(url_pattern, case=False, na=False, regex=True)
mention_mask = text_series.str.contains(mention_pattern, case=False, na=False, regex=True)
hashtag_mask = text_series.str.contains(hashtag_pattern, case=False, na=False, regex=True)
emoji_mask = text_series.str.contains(EMOJI_REGEX, regex=True, na=False)
empty_mask = text_series.str.strip() == ""
short_mask = df["n_words"] < 3
problem_counts = {
    "urls": int(url_mask.sum()),
    "mentions": int(mention_mask.sum()),
    "hashtags": int(hashtag_mask.sum()),
    "emojis": int(emoji_mask.sum()),
    "empty_text": int(empty_mask.sum()),
    "short_text": int(short_mask.sum()),
}
problem_df = pd.DataFrame({"metric": problem_counts.keys(), "count": problem_counts.values()})
problem_df["pct"] = (problem_df["count"] / len(df) * 100).round(2)
problem_df.to_csv(ARTIFACTS_DIR / "problematic_counts.csv", index=False)
print("Problematic counts saved: artifacts/problematic_counts.csv")

# Generar conteos de unigrams y bigrams con preprocesado mínimo preservando emojis.
def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^\w\s#@]", " ", text)
    return text.split()

vectorizer = CountVectorizer(tokenizer=simple_tokenize, ngram_range=(1, 1), max_features=200)
unigram_matrix = vectorizer.fit_transform(text_series)
unigram_counts = Counter({word: count for word, count in zip(vectorizer.get_feature_names_out(), unigram_matrix.sum(axis=0).A1)})
unigram_df = pd.DataFrame(unigram_counts.items(), columns=["token", "count"]).sort_values("count", ascending=False).head(20)
unigram_df.to_csv(ARTIFACTS_DIR / "top20_unigrams.csv", index=False)
print("Top unigrams saved: artifacts/top20_unigrams.csv")

vectorizer_bi = CountVectorizer(tokenizer=simple_tokenize, ngram_range=(2, 2), max_features=200)
bigram_matrix = vectorizer_bi.fit_transform(text_series)
bigram_counts = Counter({word: count for word, count in zip(vectorizer_bi.get_feature_names_out(), bigram_matrix.sum(axis=0).A1)})
bigram_df = pd.DataFrame(bigram_counts.items(), columns=["token", "count"]).sort_values("count", ascending=False).head(20)
bigram_df.to_csv(ARTIFACTS_DIR / "top20_bigrams.csv", index=False)
print("Top bigrams saved: artifacts/top20_bigrams.csv")

# Guardar figuras de los top tokens.
plt.figure(figsize=(10, 5))
sns.barplot(data=unigram_df.head(15), x="count", y="token", orient="h")
plt.title("Top 15 unigrams")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "top20_unigrams.png", dpi=200)
plt.close()

plt.figure(figsize=(10, 5))
sns.barplot(data=bigram_df.head(15), x="count", y="token", orient="h")
plt.title("Top 15 bigrams")
plt.tight_layout()
plt.savefig(ARTIFACTS_DIR / "top20_bigrams.png", dpi=200)
plt.close()

# Generar una wordcloud de los unigrams más frecuentes y, si falla con emojis, usar una versión sin emojis.
try:
    wordcloud = WordCloud(width=1200, height=700, background_color="white").generate_from_frequencies(unigram_counts)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "wordcloud.png", dpi=200)
    plt.close()
    print("Wordcloud saved: artifacts/wordcloud.png")
except Exception as exc:
    print(f"Wordcloud fallback: {exc}")
    text_for_cloud = " ".join(simple_tokenize(" ".join(text_series.tolist())))
    wordcloud = WordCloud(width=1200, height=700, background_color="white").generate(text_for_cloud)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "wordcloud.png", dpi=200)
    plt.close()
    print("Wordcloud saved with fallback: artifacts/wordcloud.png")

# Calcular top palabras por clase si existe etiqueta.
if label_column is not None and df[label_column].nunique(dropna=False) >= 2:
    class_word_counts = []
    for class_value, group in df.groupby(label_column):
        class_text = " ".join(group[text_column].fillna("").astype(str).tolist())
        tokens = simple_tokenize(class_text)
        counter = Counter(tokens)
        top10 = pd.DataFrame(counter.items(), columns=["token", "count"]).sort_values("count", ascending=False).head(10)
        top10[label_column] = class_value
        class_word_counts.append(top10)
    top_words_by_class = pd.concat(class_word_counts, ignore_index=True)
    top_words_by_class.to_csv(ARTIFACTS_DIR / "top_words_by_class.csv", index=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_words_by_class.head(20), x="count", y="token", hue=label_column)
    plt.title("Top palabras por clase")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "top_words_by_class.png", dpi=200)
    plt.close()
    print("Top words by class saved: artifacts/top_words_by_class.csv and artifacts/top_words_by_class.png")
else:
    print("No hay etiqueta suficiente para generar top_words_by_class.")

# Correlación longitud vs etiqueta si hay dos clases.
if label_column is not None and df[label_column].nunique(dropna=False) == 2:
    from scipy import stats

    class_a = df[df[label_column] == df[label_column].dropna().unique()[0]]["n_words"]
    class_b = df[df[label_column] == df[label_column].dropna().unique()[1]]["n_words"]
    statistic, p_value = stats.mannwhitneyu(class_a, class_b, alternative="two-sided")
    with (ARTIFACTS_DIR / "length_vs_label_stats.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Clase A: {df[label_column].dropna().unique()[0]}\n")
        handle.write(f"Clase B: {df[label_column].dropna().unique()[1]}\n")
        handle.write(f"Average n_words A: {class_a.mean():.3f}\n")
        handle.write(f"Average n_words B: {class_b.mean():.3f}\n")
        handle.write(f"Mann-Whitney U p-value: {p_value:.6f}\n")
    print(f"Length vs label stats saved: artifacts/length_vs_label_stats.txt (p-value: {p_value:.6f})")
else:
    print("No hay suficiente estructura para test de longitud vs etiqueta.")

# Seleccionar 5 ejemplos por clase o 10 si no hay etiqueta, separando representativos y ambiguos.
if label_column is not None and df[label_column].nunique(dropna=False) >= 2:
    selected_rows = []
    for class_value, group in df.groupby(label_column):
        representative = group.sort_values(["n_words", "n_chars"], ascending=[False, False]).head(3)
        ambiguous = group.sample(n=2, random_state=42)
        selected_rows.extend([
            {"id": row["commentId"] if "commentId" in row.index else idx, "text": row[text_column], "label": class_value, "notes": "representative"}
            for idx, row in representative.iterrows()
        ])
        selected_rows.extend([
            {"id": row["commentId"] if "commentId" in row.index else idx, "text": row[text_column], "label": class_value, "notes": "ambiguous"}
            for idx, row in ambiguous.iterrows()
        ])
    representative_df = pd.DataFrame(selected_rows)
else:
    representative_df = pd.DataFrame({
        "id": df.index,
        "text": df[text_column],
        "label": "",
        "notes": ["representative" if idx % 2 == 0 else "ambiguous" for idx in range(len(df))],
    }).head(10)
representative_df.to_csv(ARTIFACTS_DIR / "representative_examples.csv", index=False)
print("Representative examples saved: artifacts/representative_examples.csv")
print("Muestra de ejemplos representativos:")
for _, row in representative_df.head(10).iterrows():
    safe_text = str(row["text"]).encode("utf-8", "replace").decode("utf-8")
    print(f"- {row['id']} | {row['notes']} | {safe_text}")

# Crear un notebook ejecutable con el código del EDA.
nb = nbf.v4.new_notebook()
nb["cells"] = [
    nbf.v4.new_markdown_cell("# EDA del corpus UFC comments"),
    nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nimport numpy as np\nfrom sklearn.feature_extraction.text import CountVectorizer\nfrom wordcloud import WordCloud\n\nPROJECT_ROOT = Path('..').resolve()\nRAW_DIR = PROJECT_ROOT / 'data' / 'raw'\nARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'\n\ndf = pd.read_csv(RAW_DIR / 'sample_5000.csv', encoding='utf-8')\nprint(df.shape)"),
]
notebook_path = NOTEBOOKS_DIR / "01_EDA.ipynb"
with notebook_path.open("w", encoding="utf-8") as handle:
    nbf.write(nb, handle)
print(f"Notebook saved: {notebook_path}")

# Crear un resumen corto para el Google Doc con los hallazgos clave del EDA.
summary_lines = []
summary_lines.append(f"Número de registros en la submuestra: {len(df)}")
summary_lines.append(f"¿Hay etiqueta?: {'sí' if label_column is not None else 'no'}")
summary_lines.append(f"Top 5 unigrams: {', '.join(unigram_df.head(5)['token'].tolist())}")
summary_lines.append(f"Longitud media en palabras: {df['n_words'].mean():.2f}")
summary_lines.append(f"Conteos URLs: {problem_counts['urls']}; menciones: {problem_counts['mentions']}; hashtags: {problem_counts['hashtags']}; emojis: {problem_counts['emojis']}")
summary_lines.append(f"Rutas de archivos generados: {SAMPLE_FILE.relative_to(PROJECT_ROOT)}, {ARTIFACTS_DIR.relative_to(PROJECT_ROOT)}, {notebook_path.relative_to(PROJECT_ROOT)}, {SRC_DIR.relative_to(PROJECT_ROOT) / 'eda.py'}")
summary_path = ARTIFACTS_DIR / "eda_summary.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
print(f"EDA summary: {summary_path}")
print("\n".join(summary_lines))
