# Bloque de uso: ejecutar desde la raíz del proyecto:
# python src/preprocessing_step3.py
# Dependencias: nltk, spacy, en_core_web_sm, pandas
# Si faltan, instalar: pip install nltk spacy en_core_web_sm
# Y descargar recursos: python -m nltk.downloader punkt wordnet omw-1.4
# Para spaCy: python -m spacy download en_core_web_sm

# Importar librerías básicas y utilidades de tiempo y rutas.
import re
import sys
import time
from pathlib import Path
from collections import Counter

# Asegurar que la salida de consola soporte UTF-8 en Windows y reemplace caracteres no representables.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import pandas as pd

# Importar NLTK y spaCy en tiempo de ejecución para gestionar descargas si faltan.
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import wordnet

import spacy

# Definir rutas del proyecto y artefactos.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = RAW_DIR / 'sample_5000.csv'

# Reglas de limpieza (ver documentación en artifacts/regex_used.txt).
# Decisión hashtags: eliminar por defecto (documentado en el reporte).
HASHTAG_DECISION = 'eliminar'

# Regex para conservar emojis: rango básico Unicode que cubre muchos emojis.
EMOJI_REGEX = (
    '[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\u2600-\u26FF\u2700-\u27BF]'
)

# Regex principal: elimina URLs, menciones, hashtags (si se decide), y caracteres no alfanuméricos excepto emojis.
REGEX_URL = r'http\S+|www\.\S+'
REGEX_MENTION = r'@\w+'
REGEX_HASHTAG = r'#\w+'
# Para eliminar caracteres no alfanuméricos pero preservar emojis, construimos un patrón que captura todo excepto letras, números, espacios y emojis.
REGEX_NON_ALNUM_EXCEPT_EMOJI = r"[^\w\s" + EMOJI_REGEX[1:-1] + r"]+"

# Guardar la regex usada en un archivo para referencia.
(ARTIFACTS_DIR / 'regex_used.txt').write_text('\n'.join([
    'REGEX_URL: ' + REGEX_URL,
    'REGEX_MENTION: ' + REGEX_MENTION,
    'REGEX_HASHTAG: ' + REGEX_HASHTAG,
    'REGEX_NON_ALNUM_EXCEPT_EMOJI: ' + REGEX_NON_ALNUM_EXCEPT_EMOJI,
    'HASHTAG_DECISION: ' + HASHTAG_DECISION,
]), encoding='utf-8')

# Función de limpieza según las reglas indicadas en el orden especificado.
def clean_text(text):
    # Asegurar que sea string y pasar a minúsculas.
    s = '' if pd.isna(text) else str(text)
    s = s.lower()
    # Eliminar URLs.
    s = re.sub(REGEX_URL, ' ', s)
    # Eliminar menciones.
    s = re.sub(REGEX_MENTION, ' ', s)
    # Tratar hashtags según decisión (por defecto eliminar).
    if HASHTAG_DECISION == 'eliminar':
        s = re.sub(REGEX_HASHTAG, ' ', s)
    # Eliminar caracteres no alfanuméricos excepto emojis usando el patrón construido.
    s = re.sub(REGEX_NON_ALNUM_EXCEPT_EMOJI, ' ', s)
    # Normalizar espacios múltiples y recortar.
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# Mostrar y guardar ejemplos antes/después.
if not SAMPLE_FILE.exists():
    raise FileNotFoundError(f'No se encontró la submuestra: {SAMPLE_FILE}')

df_sample = pd.read_csv(SAMPLE_FILE, encoding='utf-8')

# Confirmar columna de texto (usar la heurística del paso anterior: content preferido).
text_column = None
for c in ['content', 'text', 'comment', 'body', 'message']:
    if c in df_sample.columns:
        text_column = c
        break
if text_column is None:
    # usar la primera columna object
    for c in df_sample.columns:
        if pd.api.types.is_object_dtype(df_sample[c]):
            text_column = c
            break
if text_column is None:
    raise RuntimeError('No se pudo detectar columna de texto en la muestra')

# Aplicar limpieza a toda la muestra (pero trabajaremos con los primeros 500 para lematización).
df_sample['cleaned'] = df_sample[text_column].apply(clean_text)

# Guardar antes/después de 5 ejemplos.
before_after = pd.DataFrame({
    'id': df_sample.index[:5],
    'original': df_sample[text_column].astype(str).iloc[:5].tolist(),
    'cleaned': df_sample['cleaned'].iloc[:5].tolist(),
})
before_after.to_csv(ARTIFACTS_DIR / 'clean_examples_before_after.csv', index=False, encoding='utf-8')
print('Clean examples saved: artifacts/clean_examples_before_after.csv')

# Tokenización mínima: split por espacios tras clean_text; eliminar vacíos.

def simple_tokenize(text):
    if pd.isna(text):
        return []
    tokens = [t for t in str(text).split(' ') if t]
    return tokens

sample_tokenized = df_sample['cleaned'].apply(simple_tokenize)
# Mostrar 10 ejemplos tokenizados y guardar.
token_examples = pd.DataFrame({
    'id': df_sample.index[:10],
    'tokens': sample_tokenized.iloc[:10].apply(lambda x: ' '.join(x)).tolist(),
})
token_examples.to_csv(ARTIFACTS_DIR / 'tokenized_examples.csv', index=False, encoding='utf-8')
print('Tokenized examples saved: artifacts/tokenized_examples.csv')

# Preparar NLTK: descargar recursos si es necesario.
try:
    nltk.data.find('tokenizers/punkt')
except Exception:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except Exception:
    nltk.download('wordnet')
try:
    nltk.data.find('corpora/omw-1.4')
except Exception:
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Seleccionar los mismos 500 ejemplos: primeros 500 de la submuestra.
df_500 = df_sample.iloc[:500].copy()

# Función auxiliar que usa NLTK para lematizar tokens y contar cambios.
def nltk_lemmatize(tokens):
    lemmas = []
    n_changed = 0
    for t in tokens:
        lemma = lemmatizer.lemmatize(t)
        lemmas.append(lemma)
        if lemma != t:
            n_changed += 1
    return lemmas, n_changed

# Ejecutar NLTK sobre los 500 y medir tiempo.
start = time.time()
results_nltk = []
for idx, row in df_500.iterrows():
    original = row[text_column]
    cleaned = row['cleaned']
    tokens = simple_tokenize(cleaned)
    lemmas, n_changed = nltk_lemmatize(tokens)
    results_nltk.append({
        'id': int(idx),
        'original': original,
        'cleaned': cleaned,
        'tokens_count': len(tokens),
        'lemmas': ' '.join(lemmas),
        'n_tokens_changed': n_changed,
    })
end = time.time()
elapsed_nltk = end - start
# Guardar CSV NLTK.
pd.DataFrame(results_nltk).to_csv(ARTIFACTS_DIR / 'nltk_lemmatization_500.csv', index=False, encoding='utf-8')
print(f'NLTK lemmatization time (500): {elapsed_nltk:.4f} s; total tokens changed: {sum(r["n_tokens_changed"] for r in results_nltk)}')

# Preparar spaCy: descargar modelo si hace falta y cargar.
try:
    nlp = spacy.load('en_core_web_sm')
except Exception:
    # intentar instalar y cargar
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
    nlp = spacy.load('en_core_web_sm')

# Función spaCy lemmatize
def spacy_lemmatize(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    n_changed = sum(1 for token in doc if token.lemma_ != token.text)
    return lemmas, n_changed

# Ejecutar spaCy sobre los mismos 500 y medir tiempo.
start = time.time()
results_spacy = []
for idx, row in df_500.iterrows():
    original = row[text_column]
    cleaned = row['cleaned']
    lemmas, n_changed = spacy_lemmatize(cleaned)
    tokens_count = len([t for t in cleaned.split(' ') if t])
    results_spacy.append({
        'id': int(idx),
        'original': original,
        'cleaned': cleaned,
        'tokens_count': tokens_count,
        'lemmas': ' '.join(lemmas),
        'n_tokens_changed': n_changed,
    })
end = time.time()
elapsed_spacy = end - start
pd.DataFrame(results_spacy).to_csv(ARTIFACTS_DIR / 'spacy_lemmatization_500.csv', index=False, encoding='utf-8')
print(f'spaCy lemmatization time (500): {elapsed_spacy:.4f} s; total tokens changed: {sum(r["n_tokens_changed"] for r in results_spacy)}')

# Comparativa cuantitativa y heurística simple para preferir uno u otro.
from nltk.corpus import wordnet as wn

comp_rows = []
for r_nltk, r_spacy in zip(results_nltk, results_spacy):
    idv = r_nltk['id']
    original = r_nltk['original']
    cleaned = r_nltk['cleaned']
    nltk_lemma = r_nltk['lemmas']
    spacy_lemma = r_spacy['lemmas']
    nltk_changed = r_nltk['n_tokens_changed']
    spacy_changed = r_spacy['n_tokens_changed']
    preferred = 'ninguno'
    reason = 'lemmas identical' if nltk_lemma == spacy_lemma else ''
    if nltk_lemma != spacy_lemma:
        # heurística: cuenta cuántos tokens resultantes aparecen en WordNet
        nltk_tokens = [t for t in nltk_lemma.split(' ') if t]
        spacy_tokens = [t for t in spacy_lemma.split(' ') if t]
        nltk_wn = sum(1 for t in nltk_tokens if wn.synsets(t))
        spacy_wn = sum(1 for t in spacy_tokens if wn.synsets(t))
        if nltk_wn > spacy_wn:
            preferred = 'NLTK'
            reason = f'NLTK more in WordNet ({nltk_wn}>{spacy_wn})'
        elif spacy_wn > nltk_wn:
            preferred = 'spaCy'
            reason = f'spaCy more in WordNet ({spacy_wn}>{nltk_wn})'
        else:
            # desempate por longitud promedio de tokens (más corta más lematizada)
            if sum(len(t) for t in nltk_tokens) < sum(len(t) for t in spacy_tokens):
                preferred = 'NLTK'
                reason = 'shorter average lemma length'
            elif sum(len(t) for t in spacy_tokens) < sum(len(t) for t in nltk_tokens):
                preferred = 'spaCy'
                reason = 'shorter average lemma length'
            else:
                preferred = 'ninguno'
                reason = 'tie heuristics'
    comp_rows.append({
        'id': idv,
        'original': original,
        'cleaned': cleaned,
        'nltk_lemma': nltk_lemma,
        'spacy_lemma': spacy_lemma,
        'nltk_changed_count': nltk_changed,
        'spacy_changed_count': spacy_changed,
        'preferred': preferred,
        'reason': reason,
    })

pd.DataFrame(comp_rows).to_csv(ARTIFACTS_DIR / 'lemmatization_comparison.csv', index=False, encoding='utf-8')

# Guardar tiempos y porcentajes en un archivo de texto.
total_tokens_nltk = sum(r['tokens_count'] for r in results_nltk)
changed_tokens_nltk = sum(r['n_tokens_changed'] for r in results_nltk)
perc_nltk = changed_tokens_nltk / total_tokens_nltk * 100 if total_tokens_nltk>0 else 0

total_tokens_spacy = sum(r['tokens_count'] for r in results_spacy)
changed_tokens_spacy = sum(r['n_tokens_changed'] for r in results_spacy)
perc_spacy = changed_tokens_spacy / total_tokens_spacy * 100 if total_tokens_spacy>0 else 0

(ARTIFACTS_DIR / 'lemmatization_times.txt').write_text('\n'.join([
    f'NLTK_time_s: {elapsed_nltk:.6f}',
    f'spaCy_time_s: {elapsed_spacy:.6f}',
    f'Total_tokens_nltk: {total_tokens_nltk}',
    f'Changed_tokens_nltk: {changed_tokens_nltk}',
    f'Perc_changed_nltk: {perc_nltk:.4f}',
    f'Total_tokens_spacy: {total_tokens_spacy}',
    f'Changed_tokens_spacy: {changed_tokens_spacy}',
    f'Perc_changed_spacy: {perc_spacy:.4f}',
]), encoding='utf-8')

# Stopwords y emojis: construir lista y documentar decisiones.
try:
    from nltk.corpus import stopwords
    sw = set(stopwords.words('english'))
except Exception:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    sw = set(stopwords.words('english'))
# Extender con términos del dominio.
domain_terms = {'jon','jones','ufc'}
final_stopwords = sorted(list(sw.union(domain_terms)))
(ARTIFACTS_DIR / 'stopwords_final.txt').write_text('\n'.join(final_stopwords), encoding='utf-8')
# Decisión emojis: conservarlos como tokens (documentado)
(ARTIFACTS_DIR / 'emoji_decision.txt').write_text('Se conservan emojis como tokens; no se mapearon a etiquetas textuales.', encoding='utf-8')

# Mostrar ejemplos donde eliminar stopwords cambia significativamente el token set.
examples_sw = []
for idx, row in df_500.iterrows():
    tokens = [t for t in simple_tokenize(row['cleaned']) if t]
    tokens_no_sw = [t for t in tokens if t not in final_stopwords]
    if len(tokens) >= 4 and len(tokens_no_sw) <= max(1, int(len(tokens)/2)):
        examples_sw.append({'id': int(idx), 'original_tokens': ' '.join(tokens), 'no_stopwords': ' '.join(tokens_no_sw)})
    if len(examples_sw) >= 10:
        break
pd.DataFrame(examples_sw).to_csv(ARTIFACTS_DIR / 'stopword_examples.csv', index=False, encoding='utf-8')

# Resumen final con decisiones y rutas.
summary_lines = []
summary_lines.append('regex_used: ' + str((ARTIFACTS_DIR / 'regex_used.txt').resolve()))
summary_lines.append('hashtag_decision: ' + HASHTAG_DECISION)
summary_lines.append('emoji_decision_file: ' + str((ARTIFACTS_DIR / 'emoji_decision.txt').resolve()))
summary_lines.append('stopwords_file: ' + str((ARTIFACTS_DIR / 'stopwords_final.txt').resolve()))
summary_lines.append(f'NLTK_time_s: {elapsed_nltk:.6f}')
summary_lines.append(f'spaCy_time_s: {elapsed_spacy:.6f}')
summary_lines.append(f'Perc_changed_nltk: {perc_nltk:.4f}')
summary_lines.append(f'Perc_changed_spacy: {perc_spacy:.4f}')
summary_lines.append('nltk_csv: ' + str((ARTIFACTS_DIR / 'nltk_lemmatization_500.csv').resolve()))
summary_lines.append('spacy_csv: ' + str((ARTIFACTS_DIR / 'spacy_lemmatization_500.csv').resolve()))
summary_lines.append('comparison_csv: ' + str((ARTIFACTS_DIR / 'lemmatization_comparison.csv').resolve()))
summary_lines.append('clean_examples: ' + str((ARTIFACTS_DIR / 'clean_examples_before_after.csv').resolve()))
summary_lines.append('tokenized_examples: ' + str((ARTIFACTS_DIR / 'tokenized_examples.csv').resolve()))
summary_lines.append('representative_examples: ' + str((ARTIFACTS_DIR / 'representative_examples.csv').resolve()))
summary_lines.append('stopword_examples: ' + str((ARTIFACTS_DIR / 'stopword_examples.csv').resolve()))

(ARTIFACTS_DIR / 'preprocessing_summary.txt').write_text('\n'.join(summary_lines), encoding='utf-8')
print('Preprocessing summary saved: artifacts/preprocessing_summary.txt')

# Imprimir rutas finales solicitadas para verificación.
files_to_check = [
    'regex_used.txt',
    'clean_examples_before_after.csv',
    'tokenized_examples.csv',
    'nltk_lemmatization_500.csv',
    'spacy_lemmatization_500.csv',
    'lemmatization_comparison.csv',
    'lemmatization_times.txt',
    'stopwords_final.txt',
    'emoji_decision.txt',
    'preprocessing_summary.txt',
]
for fname in files_to_check:
    p = ARTIFACTS_DIR / fname
    print(('- ' + fname + ': ') + ('OK' if p.exists() else 'MISSING'))

print('Script guardado: src/preprocessing_step3.py')
