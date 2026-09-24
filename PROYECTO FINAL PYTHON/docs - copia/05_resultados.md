# 📊 Resultados

## Modelo Seleccionado

**Regresión Logística + Calibración Isotónica**

Elegida sobre Árbol de Decisión, Random Forest y Gradient Boosting Histográfico por su balance entre rendimiento, interpretabilidad y exportabilidad.

### Hiperparámetros óptimos

| Parámetro | Valor |
|-----------|-------|
| `penalty` | l2 |
| `C` | 1.0 |
| `solver` | lbfgs |
| Calibración | Isotónica (`CalibratedClassifierCV`) |

---

## Métricas del Modelo

### Conjunto de prueba (30% de los datos)

| Métrica | Valor | Criterio | Cumple |
|---------|-------|----------|--------|
| **ROC-AUC** | 0.987 | ≥ 0.95 | ✅ |
| **Recall (sensibilidad)** | 0.946 | ≥ 0.90 | ✅ |
| **Precisión** | 0.859 | ≥ 0.80 | ✅ |
| **F1** | 0.900 | — | ✅ |
| **F2 (prioriza recall)** | 0.920 | — | ✅ |
| **Exactitud (accuracy)** | 0.941 | — | ✅ |
| **PR-AUC** | 0.975 | — | ✅ |
| **% del aula alertado** | 34.6% | ≤ 35% | ✅ |

> **Los cuatro criterios de éxito definidos en la Fase 1 se cumplen satisfactoriamente.**

---

## Umbral de Decisión

### Configuración asimétrica

El umbral **no es el 0.5 por defecto**. Se seleccionó modelando:

- **Costo asimétrico 5:1** — Un falso negativo (estudiante que reprueba sin intervención) tiene un costo 5 veces mayor que una falsa alarma.
- **Restricción operativa** — No alertar a más del 35% del aula, porque un docente no puede dar seguimiento intensivo a media clase.

### Bandas de riesgo

| Banda | Rango de probabilidad | Color | Acción |
|-------|----------------------|-------|--------|
| **BAJO** | < 0.22 | 🟢 Verde | Monitoreo mensual |
| **MEDIO** | 0.22 – 0.37 | 🟡 Ámbar | Seguimiento quincenal |
| **ALTO** | ≥ 0.37 | 🔴 Rojo | Intervención inmediata |

---

## Comparación de Modelos Candidatos

| Modelo | ROC-AUC | Recall | F2 | Selección |
|--------|---------|--------|----|-----------|
| **Regresión Logística** | **0.987** | **0.946** | **0.920** | ✅ Seleccionado |
| Random Forest | 0.985 | 0.938 | 0.915 | — |
| Gradient Boosting | 0.989 | 0.941 | 0.918 | — |
| Árbol de Decisión | 0.921 | 0.883 | 0.871 | — |

### ¿Por qué no Gradient Boosting?

Aunque Gradient Boosting obtuvo un ROC-AUC ligeramente superior (0.989 vs 0.987), la Regresión Logística fue seleccionada porque:

1. **Interpretabilidad:** Los coeficientes se pueden explicar directamente al docente.
2. **Exportabilidad:** Se puede convertir a una fórmula matemática simple (`z = intercepto + Σ coef × x`) para ejecutar en el navegador.
3. **Robustez:** Menor riesgo de sobreajuste con datos simulados.
4. **Calibración natural:** Las probabilidades son más confiables sin necesidad de calibración compleja.

---

## Variables Más Influyentes

Las variables con mayor peso en la predicción (por magnitud del coeficiente estandarizado):

1. **`calificacion_p2`** — La nota más reciente es el predictor más fuerte.
2. **`calificacion_p1`** — El historial de P1 refuerza la señal.
3. **`promedio_competencias`** — El desempeño por competencias captura habilidades transversales.
4. **`pct_asistencia`** — La asistencia es un indicador temprano de desenganche.
5. **`pct_practicas_entregadas`** — Las entregas reflejan compromiso y comprensión.
6. **`participacion`** — La participación activa se correlaciona con el éxito académico.

---

## Distribución de Riesgo

### En el conjunto de demostración (3,510 registros)

| Banda | Cantidad | Porcentaje |
|-------|----------|-----------|
| BAJO | 2,296 | 65.4% |
| MEDIO | 429 | 12.2% |
| ALTO | 785 | 22.4% |

### Tasa de riesgo real
- **29.3%** de los registros tienen `riesgo_no_promocion = 1`
- El modelo alerta al **34.6%** del aula (incluye margen de seguridad)

---

## Advertencia Metodológica

> **Importante:** Las métricas reportadas miden la capacidad del algoritmo de recuperar un proceso generador conocido (simulación); **no son evidencia de desempeño sobre estudiantes reales**. Cuando el docente cierre el Período 2, se debe reentrenar el modelo con calificaciones reales para validar su utilidad en la práctica.

---

## Evidencia Gráfica

El cuaderno genera 9 figuras que documentan:

1. Distribución de calificaciones por período
2. Matriz de correlación de variables numéricas
3. Curva ROC de los 4 modelos candidatos
4. Curva Precisión-Recall
5. Matriz de confusión con umbral óptimo
6. Calibración de probabilidades
7. Importancia de variables
8. Análisis de equidad por subgrupo
9. Distribución de bandas de riesgo

Estas figuras se muestran en la sección "Acerca del modelo" de la aplicación Streamlit.
