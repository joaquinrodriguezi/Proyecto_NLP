# 🧠 Proyecto NLP — Clasificador de Sentimiento

Clasificador de sentimiento desarrollado sobre un dataset de **5.000 comentarios**. El proyecto implementa un pipeline completo de NLP: EDA, preprocesamiento, vectorización TF-IDF, modelado en PyTorch y análisis de resultados.

---

## 🚀 Tecnologías utilizadas

- Python
- spaCy
- NLTK
- Scikit-learn
- PyTorch
- Matplotlib / Seaborn

---

## 📊 Resumen del proyecto

- Limpieza y normalización del texto con Regex
- Comparación de lematización: NLTK vs spaCy
- Lematización final con spaCy
- Representación numérica con TF-IDF (ngrams 1–2)
- Modelo `TextClassifier` (PyTorch) con:
  - Dos capas densas
  - Batch Normalization
  - Dropout (p=0.5)
- Entrenamiento con Adam + CrossEntropy + Early Stopping (patience=5)
- Análisis de errores y métricas finales

---

## 📈 Resultados principales

| Métrica         | Valor     |
|------------------|-----------|
| Accuracy         | 0.776     |
| F1 macro         | 0.756     |
| Early Stopping   | Época 7   |

**Errores analizados:** ambigüedad, sarcasmo y conflictos con emojis.

---

## 📁 Estructura del repositorio

```
src/           → scripts del pipeline completo
data/          → datos crudos y procesados
artifacts/     → modelos, métricas y gráficos
notebooks/     → documentación paso a paso del proyecto
```

---

## 🧪 Próximos pasos

- [ ] Crear un conjunto de anotaciones manuales (gold set)
- [ ] Probar reducción de dimensionalidad (TruncatedSVD)
- [ ] Evaluar modelos contextuales (DistilBERT)
- [ ] Implementar umbral de confianza para producción

---

## 📬 Autor

**Joaquín Rodríguez**
Proyecto académico / portfolio de NLP