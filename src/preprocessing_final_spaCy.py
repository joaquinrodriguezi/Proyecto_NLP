# Bloque de uso: ejecutar desde la raíz del proyecto con:
# python src/preprocessing_final.py
# Dependencias: pandas, spacy, nltk
# Si falta spaCy: python -m pip install spacy
# Si falta el modelo: python -m spacy download en_core_web_sm

import re
import sys
import time
import unicodedata
from pathlib import Path

import pandas as pd
import spacy
from sklearn.feature_extraction import _stop_words

# Forzar salida UTF-8 en Windows para evitar errores de impresión de emojis y caracteres Unicode.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

# Definir rutas principales de entrada y salida.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = RAW_DIR / 'sample_5000.csv'
CLEAN_FILE = PROCESSED_DIR / 'sample_5000_clean.csv'
LEMMATIZED_FILE = PROCESSED_DIR / 'sample_5000_lemmatized.csv'
REGEX_FILE = ARTIFACTS_DIR / 'regex_used.txt'
EMOJI_MAP_FILE = ARTIFACTS_DIR / 'emoji_map.txt'
STOPWORDS_FILE = ARTIFACTS_DIR / 'stopwords_final.txt'
TOKENIZED_NO_SW_FILE = ARTIFACTS_DIR / 'tokenized_no_stopwords_sample.csv'
CLEAN_VS_LEMMA_FILE = ARTIFACTS_DIR / 'clean_vs_lemma_examples.csv'

# Regex exacta usada para la limpieza.
URL_REGEX = r'http\S+|www\\.\\S+'
MENTION_REGEX = r'@\\w+'
HASHTAG_REGEX = r'#(\\w+)'
# Rangos Unicode de emojis seleccionados para preservar en la limpieza.
EMOJI_RANGES = (
    '\\U0001F300-\\U0001F5FF'
    '\\U0001F600-\\U0001F64F'
    '\\U0001F680-\\U0001F6FF'
    '\\U0001F700-\\U0001F77F'
    '\\U0001F780-\\U0001F7FF'
    '\\U0001F800-\\U0001F8FF'
    '\\U0001F900-\\U0001F9FF'
    '\\U0001FA00-\\U0001FA6F'
    '\\U0001FA70-\\U0001FAFF'
    '\\u2600-\\u26FF'
    '\\u2700-\\u27BF'
)
EMOJI_CLASS = '[' + EMOJI_RANGES + ']'
NON_ALNUM_EXCEPT_EMOJI_REGEX = rf'[^\w\s{EMOJI_RANGES}]+'

# Guardar decisiones y patrones en artifacts/regex_used.txt.
REGEX_LINES = [
    f'URL_REGEX: {URL_REGEX}',
    f'MENTION_REGEX: {MENTION_REGEX}',
    f'HASHTAG_REGEX: {HASHTAG_REGEX} (se elimina el símbolo # y se conserva la palabra)',
    f'NON_ALNUM_EXCEPT_EMOJI_REGEX: {NON_ALNUM_EXCEPT_EMOJI_REGEX}',
    'Hashtag decision: eliminar el símbolo # y conservar la palabra',
    'Emoji decision: preservar emojis en texto y mapearlos a etiquetas textuales antes de lematizar',
    'spaCy model: en_core_web_sm',
]
REGEX_FILE.write_text('\n'.join(REGEX_LINES), encoding='utf-8')

# Intentar cargar el CSV con utf-8 y si falla con latin-1.
encoding_used = 'utf-8'
try:
    df = pd.read_csv(SAMPLE_FILE, encoding='utf-8')
except UnicodeDecodeError:
    encoding_used = 'latin-1'
    df = pd.read_csv(SAMPLE_FILE, encoding='latin-1')
    REGEX_FILE.write_text(REGEX_FILE.read_text(encoding='utf-8') + f'\nencoding_used: {encoding_used}', encoding='utf-8')

# Confirmar columna de texto preferente.
text_column = None
for candidate in ['content', 'text', 'comment', 'body', 'message']:
    if candidate in df.columns:
        text_column = candidate
        break
if text_column is None:
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]):
            text_column = column
            break
if text_column is None:
    raise RuntimeError('No se pudo detectar una columna de texto en sample_5000.csv')

if 'id' not in df.columns:
    df.insert(0, 'id', df.index)

print(f'Shape inicial: {df.shape}')
print(f'Columna de texto: {text_column}')

# Función de limpieza final.
def clean_text(text):
    s = '' if pd.isna(text) else str(text)
    s = s.lower()
    s = re.sub(URL_REGEX, ' ', s)
    s = re.sub(MENTION_REGEX, ' ', s)
    s = re.sub(HASHTAG_REGEX, r'\1', s)
    s = re.sub(NON_ALNUM_EXCEPT_EMOJI_REGEX, ' ', s)
    s = s.replace('_', ' ')
    s = re.sub(r'\\s+', ' ', s).strip()
    return s

# Regex para detectar emojis individuales.
EMOJI_PATTERN = re.compile(
    EMOJI_CLASS,
    flags=re.UNICODE,
)

# Función que genera etiqueta a partir del nombre Unicode.
def emoji_to_label(ch):
    try:
        name = unicodedata.name(ch)
        # Normalizar nombre y generar etiqueta corta.
        label = name.lower().replace(' ', '_').replace('-', '_')
    except ValueError:
        label = f'emoji_{ord(ch):x}'
    return 'emoji_' + label

# Mapear emojis en texto con etiquetas cortas.

def map_emojis(text, emoji_map):
    if pd.isna(text):
        return ''
    result = []
    for ch in text:
        if EMOJI_PATTERN.match(ch):
            if ch not in emoji_map:
                emoji_map[ch] = emoji_to_label(ch)
            result.append(emoji_map[ch])
        else:
            result.append(ch)
    return ''.join(result)

# Aplicar limpieza y crear sample_5000_clean.csv.
start_clean = time.time()
df['text_raw'] = df[text_column].astype(str)
df['text_clean'] = df['text_raw'].apply(clean_text)
clean_time = time.time() - start_clean

df_clean = df[['id', 'text_raw', 'text_clean']].copy()
df_clean.to_csv(CLEAN_FILE, index=False, encoding='utf-8')
print(f'Tiempo limpieza: {clean_time:.4f} s')

# Mapear emojis y guardar emoji_map.txt.
emoji_map = {}
df_clean['text_emojis_mapped'] = df_clean['text_clean'].apply(lambda t: map_emojis(t, emoji_map))
count_emojis = df_clean['text_emojis_mapped'].apply(lambda t: bool(EMOJI_PATTERN.search(t))).sum()
with EMOJI_MAP_FILE.open('w', encoding='utf-8') as f:
    for emoji_char, label in sorted(emoji_map.items(), key=lambda x: x[1]):
        f.write(f'{emoji_char} -> {label}\n')
print(f'Registros con emojis: {count_emojis}')

# Cargar spaCy y lematizar de forma eficiente.
try:
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
except OSError:
    print('Modelo en_core_web_sm no encontrado. Instalar con: python -m spacy download en_core_web_sm')
    raise

start_lemma = time.time()
lemmas = []
tokens_count = []
for doc in nlp.pipe(df_clean['text_emojis_mapped'].astype(str), batch_size=64, n_process=1):
    tokens = [t.lemma_ for t in doc if not t.is_space]
    lemmas.append(' '.join(tokens))
    tokens_count.append(len(tokens))
lemma_time = time.time() - start_lemma

# Guardar version lematizada.
df_lemmatized = pd.DataFrame({
    'id': df_clean['id'],
    'text_clean': df_clean['text_clean'],
    'text_emojis_mapped': df_clean['text_emojis_mapped'],
    'text_lemma': lemmas,
    'n_tokens': tokens_count,
})
df_lemmatized.to_csv(LEMMATIZED_FILE, index=False, encoding='utf-8')
print(f'Tiempo lematización spaCy: {lemma_time:.4f} s')

# Stopwords final.
stopwords = set(_stop_words.ENGLISH_STOP_WORDS)
domain_stopwords = {'jon', 'jones', 'ufc'}
all_stopwords = sorted(stopwords.union(domain_stopwords))
STOPWORDS_FILE.write_text('\n'.join(all_stopwords), encoding='utf-8')

# Tokenización sin stopwords para 100 ejemplos.

sample_100 = df_lemmatized.head(100).copy()

sample_100['tokens_no_stopwords'] = sample_100['text_lemma'].apply(
    lambda text: ' '.join([t for t in text.split() if t not in all_stopwords])
)

sample_100[['id', 'text_lemma', 'tokens_no_stopwords']].to_csv(
    TOKENIZED_NO_SW_FILE, index=False, encoding='utf-8'
)

# Guardar 5 ejemplos before/after.
clean_vs_lemma = df_lemmatized[['id', 'text_clean', 'text_lemma']].head(5).copy()
clean_vs_lemma['text_raw'] = df_clean['text_raw'].head(5).values
clean_vs_lemma.to_csv(CLEAN_VS_LEMMA_FILE, index=False, encoding='utf-8')

# Imprimir ejemplos representativos.
print('Ejemplos representativos:')
for _, row in clean_vs_lemma.iterrows():
    print(f"id={row['id']} raw={row['text_raw'][:50]!r} clean={row['text_clean'][:50]!r} lemma={row['text_lemma'][:50]!r}")

print('\nArchivos generados:')
print(f' - {CLEAN_FILE}')
print(f' - {LEMMATIZED_FILE}')
print(f' - {REGEX_FILE}')
print(f' - {EMOJI_MAP_FILE}')
print(f' - {STOPWORDS_FILE}')
print(f' - {TOKENIZED_NO_SW_FILE}')
print(f' - {CLEAN_VS_LEMMA_FILE}')
print(f' - {Path(__file__)}')
print(f'Shape final clean: {df_clean.shape}, Shape final lemmatized: {df_lemmatized.shape}')
