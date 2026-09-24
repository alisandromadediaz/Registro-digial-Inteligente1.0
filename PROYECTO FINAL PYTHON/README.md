# 🎓 Registro Pedagógico Inteligente (RPI)

**Modelo de Machine Learning supervisado para la alerta temprana de no promoción**
Metodología **CRISP-ML(Q)** · Sistema Educativo Dominicano (MINERD) · Año escolar 2026-2027

Fuente de datos: `3RO_B_REGISTRO_ESCOLAR_2026_2027.xlsx` (libro de registro del docente).

---

## 1. Qué hace este proyecto

En el Nivel Secundario dominicano el docente ya registra todo lo necesario: calificaciones
por período, prácticas por competencia y asistencia diaria. El problema es que esa
información **se consulta cuando ya es tarde**: el estudiante llega a la prueba completiva
o extraordinaria sin que nadie haya intervenido a tiempo.

Este sistema toma **sólo lo observable al cierre del Período 2** y estima la probabilidad
de que un estudiante **no alcance los 70 puntos** en la calificación final ordinaria de una
asignatura, la traduce en una banda de riesgo (BAJO / MEDIO / ALTO) y entrega un **plan de
acompañamiento concreto**.

| Elemento | Definición |
|---|---|
| Tipo de aprendizaje | Supervisado |
| Tarea | Clasificación binaria |
| Unidad de análisis | Par *(estudiante, asignatura)* dentro de un año escolar |
| Variable objetivo | `riesgo_no_promocion` = 1 si C.F. ordinaria < 70 |
| Momento de predicción | Cierre del Período 2 (mitad del año) |

---

## 2. Estructura del proyecto

```
registro_pedagogico_ia/
├── notebooks/
│   └── CRISP_ML_Registro_Pedagogico.ipynb   ← Cuaderno principal (60 celdas, ya ejecutado)
├── src/
│   ├── registro_ia.py        Extracción del Excel, reglas MINERD, simulación, recomendaciones
│   └── despliegue.py         Exporta el modelo a JSON y construye la landing page
├── app/
│   └── streamlit_app.py      Panel del docente (3 secciones)
├── landing/
│   ├── index.html            Landing autónoma con el modelo embebido  ← ÁBRELA CON DOBLE CLIC
│   └── _plantilla_index.html Plantilla (el cuaderno la rellena)
├── data/
│   ├── raw/                  Libro de registro escolar original
│   └── processed/            dataset_registro_escolar.csv (3.510 registros)
├── models/
│   ├── modelo_riesgo.joblib  Pipeline completo (preproceso + clasificador calibrado)
│   ├── model_card.json       Tarjeta del modelo: métricas, supuestos y limitaciones
│   └── modelo_navegador.json Coeficientes exportados para inferencia en el navegador
├── reports/
│   ├── figuras/              9 gráficas generadas por el cuaderno
│   ├── alertas_periodo2_2026_2027.csv
│   └── bitacora_monitoreo.json
├── requirements.txt          Dependencias de la app Streamlit
└── requirements-dev.txt      Dependencias adicionales del cuaderno
```

---

## 3. Cómo ejecutarlo

### 3.1 La landing page (no requiere instalar nada)

Abre `landing/index.html` con doble clic en cualquier navegador. Funciona **sin internet y
sin servidor**: los coeficientes del modelo están dentro del archivo y el cálculo se hace
en el navegador, así que **ningún dato del estudiante sale del equipo**.

### 3.2 La aplicación Streamlit

```bash
python -m venv .venv
source .venv/bin/activate          # Windows:  .venv\Scripts\activate
pip install -r requirements.txt

streamlit run app/streamlit_app.py
```

Se abre en `http://localhost:8501` con tres secciones:

1. **Evaluación individual** — formulario por estudiante, con banda de riesgo, plan de
   acompañamiento y proyección de la calificación según las reglas del MINERD.
2. **Análisis por lote** — sube un CSV/XLSX (o usa el conjunto de demostración) y obtén el
   reporte de alertas de un curso completo, filtrable y descargable.
3. **Acerca del modelo** — tarjeta del modelo, métricas y las 9 figuras del cuaderno.

> Para desplegar en **Streamlit Community Cloud**: sube el repositorio completo y apunta a
> `app/streamlit_app.py`. El `requirements.txt` de la raíz es suficiente.

### 3.3 El cuaderno CRISP-ML(Q)

```bash
pip install -r requirements.txt -r requirements-dev.txt
jupyter lab notebooks/CRISP_ML_Registro_Pedagogico.ipynb
```

El cuaderno se entrega **ya ejecutado** (con todas las salidas y gráficas visibles). Si lo
vuelves a ejecutar de principio a fin, regenera por sí solo el dataset, el modelo, la
tarjeta, las figuras, el reporte de alertas **y la landing page**.

---

## 4. Las seis fases de CRISP-ML(Q) en el cuaderno

| Fase | Sección | Contenido |
|---|---|---|
| 1. Comprensión del negocio y de los datos | §1 – §2 | Contexto MINERD, criterios de éxito, riesgos éticos, EDA |
| 2. Preparación de datos | §3 | Control de fuga de datos, diccionario, partición agrupada, pipeline |
| 3. Modelado | §4 | 4 modelos candidatos, `GridSearchCV` con F2, calibración isotónica |
| 4. Evaluación y aseguramiento | §5 | Métricas, umbral por costo, explicabilidad, auditoría de equidad, análisis de errores |
| 5. Despliegue | §6 | `.joblib`, tarjeta del modelo, landing autónoma, reporte de alertas |
| 6. Monitoreo y mantenimiento | §7 | Deriva con PSI, plan de reentrenamiento, bitácora de auditoría |

---

## 5. Resultados obtenidos

**Modelo seleccionado:** Regresión Logística + calibración isotónica
(elegida sobre Árbol de Decisión, Random Forest y Gradient Boosting Histográfico).

| Métrica (conjunto de prueba) | Valor |
|---|---|
| ROC-AUC | 0.987 |
| Recall (sensibilidad) | 0.946 |
| Precisión | 0.859 |
| F1 | 0.900 |
| % del aula alertado | 34.6 % |

**Umbral de decisión: 0.37** (BAJO < 0.22 ≤ MEDIO < 0.37 ≤ ALTO). No es el 0.5 por
defecto: se eligió modelando un costo asimétrico **5:1** (un falso negativo es un
estudiante que reprueba sin que nadie interviniera) **sujeto a una restricción operativa**
—no alertar a más del 35 % del aula, porque un docente no puede dar seguimiento intensivo
a media clase—.

Los cuatro criterios de éxito definidos en la Fase 1 se cumplen.

---

## 6. Advertencia metodológica importante

El libro entregado corresponde al año escolar 2026-2027 y **sus celdas de calificación
están vacías** (el año apenas comenzó: todas las notas son `0` y las fórmulas devuelven
`#DIV/0!`). **No hay etiquetas reales para entrenar.**

La decisión tomada —documentada en la §2.3 del cuaderno— fue simular un histórico
**calibrado sobre la estructura real del centro**: los 195 estudiantes reales, sus 6 cursos,
las 20 asignaturas del catálogo y las reglas de promoción del MINERD. Lo simulado es
únicamente el *desempeño*, mediante un proceso generador explícito:

```text
habilidad_latente ~ Normal(0, 1)
asistencia        = f(habilidad, apoyo familiar, sobreedad) + ruido
nota_periodo      = g(habilidad, asistencia, entregas, participación, apoyo, dificultad) + ruido
C.F.              = promedio(P1..P4)
y                 = 1 si C.F. < 70
```

**Consecuencia:** las métricas miden la capacidad del algoritmo de recuperar un proceso
generador conocido; **no son evidencia de desempeño sobre estudiantes reales**.

**Para usarlo con datos reales:** cuando el docente cierre el Período 2, sustituye la
llamada a `rpi.simular_historico()` por la lectura de las hojas `Calificación`, `Prácticas`
y `Asistencia`. Todo el resto del cuaderno, la app y la landing funcionan sin cambios —ése
es el motivo de haber fijado un contrato de datos en `src/registro_ia.py`.

---

## 7. Decisiones éticas

* **Variables excluidas del entrenamiento:** nombre, sexo y nacionalidad. No aportan
  capacidad predictiva legítima y su uso convertiría el sistema en un mecanismo de
  discriminación estructural.
* **Auditoría de equidad** por grado, tanda y nivel de apoyo familiar (§5.4 del cuaderno).
* **El estudiante nunca ve su puntaje.** La salida es una *alerta de acompañamiento*, no
  una *predicción de fracaso*: evitar la profecía autocumplida es parte del diseño.
* **La decisión siempre es del docente.** El sistema es apoyo, nunca veredicto automático.
* **Privacidad:** la landing page no envía nada a ningún servidor.

---

## 8. Robustez ante errores de tipo (`TypeError`)

Requisito explícito del proyecto. Se resolvió en el diseño, no con parches:

1. **`rpi.num()` y `rpi.texto()`** — toda entrada proveniente de Excel, de un formulario
   HTML o de un CSV pasa por ahí antes de usarse. Probado con `"58,5"`, `None`, `"72%"`,
   `#DIV/0!`, `numpy.int64` y categorías inexistentes: ninguno rompe la ejecución.
2. **`crear_onehot()`** — absorbe el cambio de `sparse` a `sparse_output` en
   scikit-learn 1.2, para que el cuaderno corra en versiones anteriores y posteriores.
3. **Streamlit** — todos los `number_input` reciben `float` en `min/max/value/step` (mezclar
   `int` y `float` es la causa más común de errores de tipo en Streamlit), y un helper lee
   la firma real de `st.dataframe` para elegir entre `width="stretch"` y
   `use_container_width` según la versión instalada.
4. **Degradación elegante** — si falta `modelo_riesgo.joblib`, la app usa la especificación
   JSON como respaldo en lugar de caerse; si falta el Excel, usa valores por defecto.

Verificación realizada: el cuaderno completo (60 celdas) ejecuta sin una sola celda en
error; las tres páginas de la app se ejecutan sin excepciones; y el JavaScript de la
landing reproduce la predicción de scikit-learn con una diferencia máxima de **2.2 × 10⁻¹⁹**
sobre 50 casos de prueba.

---

## 9. Cómo continuar

* Sustituir la simulación por las calificaciones reales al cerrar el Período 2.
* Incorporar el registro de la app al conjunto de entrenamiento (aprendizaje continuo).
* Añadir explicaciones locales con SHAP para justificar cada alerta individual.
* Medir el impacto real: comparar la tasa de promoción de cursos con y sin el sistema
  durante un año escolar completo.

---

*Proyecto de aula · Machine Learning supervisado con metodología CRISP-ML(Q) ·
Sistema Educativo Dominicano*
