# 🎯 Objetivos del Proyecto

## Objetivo General

Desarrollar un **sistema de alerta temprana** basado en Machine Learning supervisado que, utilizando los indicadores del registro escolar observables al cierre del Período 2, estime la probabilidad de no promoción de cada estudiante del Nivel Secundario dominicano y genere planes de acompañamiento pedagógico personalizados, siguiendo la metodología CRISP-ML(Q).

---

## Objetivos Específicos

### 1. Comprensión del negocio y de los datos
- Analizar el sistema de evaluación y promoción del MINERD para el Nivel Secundario.
- Identificar las variables predictoras disponibles en el libro de registro del docente.
- Definir criterios de éxito medibles para el modelo predictivo.
- Documentar los riesgos éticos y las restricciones operativas del sistema.

### 2. Preparación de datos
- Extraer y transformar la matrícula real del libro de registro escolar (.xlsx).
- Construir un dataset calibrado sobre la estructura real del centro educativo.
- Implementar un contrato de datos que garantice la integridad de las variables.
- Diseñar un pipeline de preprocesamiento que prevenga la fuga de datos temporales.

### 3. Modelado
- Entrenar y comparar al menos 4 modelos candidatos (Regresión Logística, Árbol de Decisión, Random Forest, Gradient Boosting).
- Optimizar hiperparámetros mediante `GridSearchCV` con la métrica F2 (prioriza recall).
- Calibrar las probabilidades del modelo seleccionado para garantizar confiabilidad.
- Seleccionar un umbral de decisión basado en el costo asimétrico 5:1.

### 4. Evaluación y aseguramiento de calidad
- Verificar que el modelo cumple los 4 criterios de éxito definidos en la Fase 1.
- Realizar auditoría de equidad por grado, tanda y nivel de apoyo familiar.
- Analizar los errores del modelo para identificar patrones de fallo.
- Generar una tarjeta del modelo (Model Card) con métricas, supuestos y limitaciones.

### 5. Despliegue
- Desarrollar una **aplicación Streamlit** con panel del docente (evaluación individual y por lote).
- Crear una **landing page autónoma** que funcione sin servidor ni internet.
- Exportar los coeficientes del modelo a JSON para inferencia en el navegador.
- Preparar el proyecto para despliegue en **Streamlit Community Cloud**.

### 6. Monitoreo y mantenimiento
- Implementar detección de deriva de datos con el índice PSI.
- Definir un plan de reentrenamiento periódico.
- Crear una bitácora de auditoría para el seguimiento del modelo en producción.

---

## Criterios de Éxito

| # | Criterio | Meta | Resultado |
|---|----------|------|-----------|
| 1 | **Recall (sensibilidad)** ≥ 0.90 | Detectar al menos 9 de cada 10 estudiantes en riesgo real | ✅ 0.946 |
| 2 | **Precisión** ≥ 0.80 | No más de 2 falsas alarmas por cada 10 alertas | ✅ 0.859 |
| 3 | **% del aula alertado** ≤ 35% | Un docente no puede dar seguimiento intensivo a media clase | ✅ 34.6% |
| 4 | **ROC-AUC** ≥ 0.95 | Capacidad discriminativa alta del modelo | ✅ 0.987 |

> Los cuatro criterios de éxito se cumplen satisfactoriamente.
