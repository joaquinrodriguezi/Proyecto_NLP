# Bloque de uso: ejecutar este script desde la raíz del proyecto con:
# python src/load_and_inspect.py
# Este script carga el CSV de comentarios, verifica su estructura y genera artefactos en data/raw/.

# Importar las bibliotecas necesarias para manejo de rutas, expresiones regulares y análisis tabular.
import re
import sys
from pathlib import Path

import pandas as pd

# Definir la ruta raíz del proyecto para localizar carpetas y archivos de forma robusta.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
INPUT_FILE = RAW_DIR / "UFC_champion_Jon_Jones_retires_comments.csv"

# Crear las carpetas requeridas si aún no existen para no sobrescribir contenido previo.
for directory in [RAW_DIR, PROCESSED_DIR, NOTEBOOKS_DIR, SRC_DIR, ARTIFACTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Directorio listo: {directory}")

# Intentar cargar el CSV con encoding utf-8 y, si falla, con latin-1 para documentar la elección.
encoding_used = None
last_error = None
df = None
for encoding_name in ["utf-8", "latin-1"]:
    try:
        df = pd.read_csv(INPUT_FILE, encoding=encoding_name)
        encoding_used = encoding_name
        print(f"CSV cargado con encoding: {encoding_name}")
        break
    except Exception as exc:
        last_error = exc
        print(f"Fallo con encoding {encoding_name}: {exc}")

# Si no se pudo leer el archivo con ninguno de los encodings, detener la ejecución claramente.
if df is None:
    raise RuntimeError(f"No se pudo leer el archivo CSV: {INPUT_FILE} ({last_error})")

# Mostrar la forma del dataset para verificar tamaño y carga correcta.
print(f"Shape del dataset: {df.shape}")

# Detener si el dataset tiene menos de 2000 filas según la instrucción del proyecto.
if df.shape[0] < 2000:
    print("Advertencia: el dataset tiene menos de 2000 filas; se detiene la inspección.")
    sys.exit(0)

# Mostrar el listado de columnas y una vista previa de las primeras filas del dataframe.
print("Columnas detectadas:")
print(df.columns.tolist())
print("Primeras 10 filas:")
print(df.head(10).to_string(index=False))

# Guardar un sample con las primeras 100 filas en data/raw/ sin modificar el archivo original.
sample_head_path = RAW_DIR / "sample_head.csv"
if not sample_head_path.exists():
    df.head(100).to_csv(sample_head_path, index=False)
    print(f"Archivo guardado: {sample_head_path}")
else:
    print(f"El archivo ya existe y no se sobrescribe: {sample_head_path}")

# Calcular nulos por columna y el porcentaje correspondiente para la inspección inicial.
null_counts = df.isnull().sum()
null_pct = (null_counts / len(df) * 100).round(2)
print("Nulos por columna:")
for column_name in df.columns:
    print(f"{column_name} | {null_counts[column_name]} | {null_pct[column_name]}%")

# Detectar una columna de texto buscándola por nombres comunes y, si no existe, elegir la primera columna object.
text_candidates = [
    column_name
    for column_name in df.columns
    if any(keyword in column_name.lower() for keyword in ["content", "text", "comment", "body", "message"])
]
text_column = None
for candidate in ["content", "text", "comment", "body", "message"]:
    matching_column = next((column_name for column_name in df.columns if column_name.lower() == candidate), None)
    if matching_column is not None:
        text_column = matching_column
        break
if text_column is None and text_candidates:
    text_column = text_candidates[0]
if text_column is None:
    for column_name in df.columns:
        if pd.api.types.is_object_dtype(df[column_name]):
            text_column = column_name
            
            break

# Detectar una columna de etiqueta buscándola por nombres comunes y, si existe, mostrar su distribución.
label_candidates = [
    column_name
    for column_name in df.columns
    if any(keyword in column_name.lower() for keyword in ["label", "sentiment", "class", "category"])
]
label_column = label_candidates[0] if label_candidates else None
print(f"Columna de texto sugerida: {text_column}")
if label_column is not None:
    print(f"Columna de etiqueta detectada: {label_column}")
    print(df[label_column].value_counts(dropna=False).to_string())
else:
    print("No se detectó columna de etiqueta; se omite la muestra balanceada.")

# Crear una muestra balanceada por clase si existe una etiqueta con al menos dos clases diferentes.
sample_balanced_path = RAW_DIR / "sample_balanced.csv"
if label_column is not None and df[label_column].nunique(dropna=False) >= 2:
    if not sample_balanced_path.exists():
        balanced_frames = []
        for class_value in df[label_column].dropna().unique():
            subset = df[df[label_column] == class_value].head(100)
            balanced_frames.append(subset)
        balanced_df = pd.concat(balanced_frames, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        balanced_df.to_csv(sample_balanced_path, index=False)
        print(f"Archivo guardado: {sample_balanced_path}")
    else:
        print(f"El archivo ya existe y no se sobrescribe: {sample_balanced_path}")
else:
    print("No se creó sample_balanced.csv porque no existe una etiqueta válida con al menos dos clases.")

# Preparar la columna de texto para calcular estadísticas de longitud en caracteres y palabras.
text_series = df[text_column].fillna("").astype(str)
char_lengths = text_series.str.len()
word_lengths = text_series.str.split().str.len()

# Mostrar estadísticas resumidas de longitud para caracteres y palabras.
length_stats = pd.DataFrame(
    {
        "metric": ["char_length", "word_count"],
        "min": [char_lengths.min(), word_lengths.min()],
        "25%": [char_lengths.quantile(0.25), word_lengths.quantile(0.25)],
        "median": [char_lengths.median(), word_lengths.median()],
        "mean": [char_lengths.mean(), word_lengths.mean()],
        "75%": [char_lengths.quantile(0.75), word_lengths.quantile(0.75)],
        "max": [char_lengths.max(), word_lengths.max()],
    }
)
print("Estadísticas de longitud:")
print(length_stats.to_string(index=False))

# Contar registros con URLs, menciones, hashtags y textos vacíos o muy cortos para detectar ruido.
url_pattern = r"https?://|www\."
mention_pattern = r"(?<!\w)@\w+"
hashtag_pattern = r"(?<!\w)#\w+"
url_mask = text_series.str.contains(url_pattern, case=False, na=False, regex=True)
mention_mask = text_series.str.contains(mention_pattern, case=False, na=False, regex=True)
hashtag_mask = text_series.str.contains(hashtag_pattern, case=False, na=False, regex=True)
empty_mask = text_series.str.strip() == ""
short_mask = word_lengths < 3
problem_counts = {
    "urls": int(url_mask.sum()),
    "mentions": int(mention_mask.sum()),
    "hashtags": int(hashtag_mask.sum()),
    "empty_text": int(empty_mask.sum()),
    "short_text": int(short_mask.sum()),
}
problem_pct = {key: round(value / len(df) * 100, 2) for key, value in problem_counts.items()}
print("Conteos de elementos problemáticos:")
for key in problem_counts:
    print(f"{key} | {problem_counts[key]} | {problem_pct[key]}%")

# Crear un reporte de inspección en texto plano con los hallazgos más importantes.
inspection_report_path = RAW_DIR / "inspection_report.txt"
if not inspection_report_path.exists():
    report_lines = []
    report_lines.append(f"Shape del dataset: {df.shape}")
    report_lines.append("Columnas:")
    report_lines.extend([f"- {column_name}" for column_name in df.columns])
    report_lines.append("Nulos por columna:")
    for column_name in df.columns:
        report_lines.append(f"- {column_name}: {null_counts[column_name]} ({null_pct[column_name]}%)")
    report_lines.append(f"Columna de texto elegida: {text_column}")
    report_lines.append(f"Columna de etiqueta detectada: {label_column}")
    if label_column is not None:
        report_lines.append("Value counts de etiqueta:")
        report_lines.append(df[label_column].value_counts(dropna=False).to_string())
    report_lines.append("Estadísticas de longitud:")
    report_lines.append(length_stats.to_string(index=False))
    report_lines.append("Conteos de elementos problemáticos:")
    for key in problem_counts:
        report_lines.append(f"- {key}: {problem_counts[key]} ({problem_pct[key]}%)")
    report_lines.append("Rutas de archivos guardados:")
    report_lines.append(f"- {sample_head_path.relative_to(PROJECT_ROOT)}")
    if sample_balanced_path.exists():
        report_lines.append(f"- {sample_balanced_path.relative_to(PROJECT_ROOT)}")
    report_lines.append(f"- {inspection_report_path.relative_to(PROJECT_ROOT)}")
    inspection_report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Archivo guardado: {inspection_report_path}")
else:
    print(f"El archivo ya existe y no se sobrescribe: {inspection_report_path}")

# Imprimir un resumen final con las rutas de los artefactos generados para la evidencia del proyecto.
print("Resumen final:")
print(f"- Shape del dataset: {df.shape}")
print(f"- Columnas detectadas: {list(df.columns)}")
print(f"- Columna de texto sugerida: {text_column}")
print(f"- Columna de etiqueta detectada: {label_column}")
print(f"- Archivos guardados: {sample_head_path.relative_to(PROJECT_ROOT)}, {inspection_report_path.relative_to(PROJECT_ROOT)}")
if sample_balanced_path.exists():
    print(f"- Archivo adicional guardado: {sample_balanced_path.relative_to(PROJECT_ROOT)}")
print(f"- Script guardado: {SRC_DIR.relative_to(PROJECT_ROOT) / 'load_and_inspect.py'}")
