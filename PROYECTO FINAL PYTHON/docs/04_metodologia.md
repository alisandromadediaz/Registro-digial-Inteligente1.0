# 🔬 Metodología

## Aplicación de CRISP-ML(Q) al Proyecto

Este proyecto implementa las seis fases completas de la metodología CRISP-ML(Q), documentadas en el cuaderno `CRISP_ML_Registro_Pedagogico.ipynb` (60 celdas, ya ejecutado).

---

## Fase 1: Comprensión del Negocio y de los Datos (§1–§2)

### Contexto MINERD
- Se analizó el sistema de evaluación del Nivel Secundario dominicano.
- Se identificaron las reglas de promoción (ordinaria, completiva, extraordinaria).
- Se definieron 4 criterios de éxito medibles.

### Riesgos éticos documentados
- Discriminación estructural por uso de variables demográficas.
- Profecía autocumplida si el estudiante conoce su "puntuación".
- Dependencia excesiva del sistema por parte del docente.

### Análisis Exploratorio de Datos (EDA)
- Matrícula: 195 estudiantes en 6 cursos.
- Catálogo: 20 asignaturas del currículo.
- Dataset generado: 3,510 registros (par estudiante-asignatura × 3 años).
- Tasa de riesgo natural: 29.3%.

---

## Fase 2: Preparación de Datos (§3)

### Control de fuga de datos
Se verificó que ninguna variable de entrada contenga información del futuro:
- Solo se usan datos observables al **cierre del Período 2**.
- Las calificaciones de P3 y P4 se excluyen del modelo.
- La variable objetivo se calcula con la C.F. ordinaria (promedio de los 4 períodos).

### Diccionario de variables

| Variable | Tipo | Rango | Descripción |
|----------|------|-------|-------------|
| `calificacion_p1` | Numérica | 0-100 | Nota del Período 1 |
| `calificacion_p2` | Numérica | 0-100 | Nota del Período 2 |
| `promedio_competencias` | Numérica | 0-100 | Promedio C1-C4 |
| `pct_asistencia` | Numérica | 0-100 | % de asistencia acumulada |
| `tardanzas` | Numérica | 0-60 | Cantidad de tardanzas |
| `ausencias_injustificadas` | Numérica | 0-60 | Ausencias sin justificación |
| `pct_practicas_entregadas` | Numérica | 0-100 | % de actividades entregadas |
| `participacion` | Numérica | 0-10 | Participación en clase |
| `sobreedad_anios` | Numérica | 0-4 | Años de sobreedad |
| `recuperaciones_p1_p2` | Numérica | 0-2 | Recuperaciones en P1-P2 |
| `horas_estudio_semana` | Numérica | 0-20 | Horas de estudio semanal |
| `nivel` | Categórica | 4to/5to/6to | Grado académico |
| `asignatura` | Categórica | 20 opciones | Materia evaluada |
| `apoyo_familiar` | Categórica | Bajo/Medio/Alto | Nivel de acompañamiento |
| `tanda` | Categórica | Mat./Vesp./Ext. | Turno escolar |

### Partición agrupada
```
Entrenamiento: 70% (agrupado por estudiante para evitar fuga)
Prueba: 30%
```

### Pipeline de preprocesamiento
```python
ColumnTransformer([
    ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                      ("escalar", StandardScaler())]),
     COLUMNAS_NUMERICAS),
    ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                      ("codificar", OneHotEncoder(handle_unknown="ignore"))]),
     COLUMNAS_CATEGORICAS),
])
```

---

## Fase 3: Modelado (§4)

### Modelos candidatos
Se evaluaron 4 algoritmos mediante `GridSearchCV` con validación cruzada estratificada (5 folds) y la métrica F2:

1. **Regresión Logística** (penalty: l1, l2; C: 0.01-100)
2. **Árbol de Decisión** (max_depth: 3-15; min_samples_leaf: 5-50)
3. **Random Forest** (n_estimators: 50-300; max_depth: 5-20)
4. **HistGradientBoosting** (max_iter: 50-300; max_depth: 3-10)

### Modelo seleccionado
**Regresión Logística** con calibración isotónica fue elegida por:
- Mejor balance entre rendimiento y interpretabilidad.
- Coeficientes directamente explicables para el docente.
- Probabilidades bien calibradas tras calibración isotónica.
- Exportable a JSON para inferencia en el navegador.

### Calibración isotónica
Las probabilidades del modelo se recalibraron con `CalibratedClassifierCV(method="isotonic")` para garantizar que una predicción de "60% de riesgo" corresponda realmente a un 60% de casos positivos.

---

## Fase 4: Evaluación y Aseguramiento (§5)

### Métricas en el conjunto de prueba

| Métrica | Valor |
|---------|-------|
| **ROC-AUC** | 0.987 |
| **Recall (sensibilidad)** | 0.946 |
| **Precisión** | 0.859 |
| **F1** | 0.900 |
| **F2** | 0.920 |
| **% del aula alertado** | 34.6% |

### Umbral óptimo por costo
- Costo de falso negativo: 5× (estudiante no detectado)
- Restricción: ≤ 35% del aula alertado
- **Umbral resultante: 0.37**

### Auditoría de equidad
Se verificó que recall y precisión no varían significativamente entre:
- Grados: 4to, 5to, 6to
- Tandas: Matutina, Vespertina, Extendida
- Apoyo familiar: Bajo, Medio, Alto

### Análisis de errores
Se estudiaron los falsos negativos y falsos positivos para identificar patrones de fallo y establecer los límites del modelo.

---

## Fase 5: Despliegue (§6)

### Artefactos generados

| Artefacto | Formato | Propósito |
|-----------|---------|-----------|
| `modelo_riesgo.joblib` | Pipeline serializado | Inferencia en la app Streamlit |
| `model_card.json` | JSON | Tarjeta del modelo (métricas, limitaciones) |
| `modelo_navegador.json` | JSON | Coeficientes para inferencia en el navegador |
| `index.html` | HTML autónomo | Landing page sin servidor |
| `streamlit_app.py` → `app.py` | Python | Panel del docente |
| `alertas_periodo2.csv` | CSV | Reporte de alertas por lote |

### Verificación de equivalencia
Se verificó que la inferencia en JavaScript reproduce la predicción de scikit-learn con una diferencia máxima de **2.2 × 10⁻¹⁹** sobre 50 casos de prueba.

---

## Fase 6: Monitoreo y Mantenimiento (§7)

### Detección de deriva
- **PSI (Population Stability Index):** Se calcula periódicamente para detectar cambios en la distribución de las variables de entrada.
- **Umbral de alerta:** PSI > 0.25 indica deriva significativa.

### Plan de reentrenamiento
1. Al cierre de cada Período 2, recopilar calificaciones reales.
2. Reentrenar el modelo con los datos acumulados.
3. Recalcular métricas y comparar con la línea base.
4. Actualizar la tarjeta del modelo y la landing page.

### Bitácora de auditoría
Se mantiene un registro JSON con cada actualización del modelo, incluyendo:
- Fecha de entrenamiento
- Versión del modelo
- Métricas obtenidas
- Cambios en los datos
