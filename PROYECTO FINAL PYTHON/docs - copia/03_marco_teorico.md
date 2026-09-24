# 📚 Marco Teórico

## 1. CRISP-ML(Q): Metodología del Proyecto

### ¿Qué es CRISP-ML(Q)?

**CRISP-ML(Q)** (*Cross Industry Standard Process for the development of Machine Learning applications with Quality assurance*) es una extensión del modelo CRISP-DM diseñada específicamente para proyectos de Machine Learning. Fue propuesta por Studer et al. (2021) y añade aseguramiento de calidad (Quality assurance) a cada fase del proceso.

### Las seis fases

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Comprensión       2. Preparación     3. Modelado           │
│     del Negocio          de Datos            (Entrenamiento)    │
│         ↓                   ↓                    ↓              │
│  4. Evaluación        5. Despliegue      6. Monitoreo          │
│     y Aseguramiento      (Deployment)       y Mantenimiento    │
└─────────────────────────────────────────────────────────────────┘
```

| Fase | Objetivo | Artefacto principal |
|------|----------|---------------------|
| 1. Comprensión del negocio | Definir el problema, criterios de éxito y riesgos | Documento de requisitos |
| 2. Preparación de datos | Limpiar, transformar y particionar los datos | Dataset y pipeline |
| 3. Modelado | Entrenar y comparar algoritmos candidatos | Modelo entrenado |
| 4. Evaluación | Validar métricas, equidad y explicabilidad | Tarjeta del modelo |
| 5. Despliegue | Poner el modelo en producción | Aplicación web |
| 6. Monitoreo | Detectar degradación y planificar reentrenamiento | Bitácora de auditoría |

### Ventaja clave sobre CRISP-DM

CRISP-ML(Q) incorpora **controles de calidad explícitos** en cada fase:
- Restricciones de equidad y ética
- Verificación de la equivalencia numérica tras la exportación
- Detección de deriva de datos (data drift) en producción
- Documentación mediante Model Cards

---

## 2. Machine Learning Supervisado

### Definición

El **aprendizaje supervisado** es un paradigma de Machine Learning donde el algoritmo aprende una función de mapeo `f: X → y` a partir de un conjunto de pares de entrenamiento `(x_i, y_i)`, donde:
- `X` son las variables de entrada (features)
- `y` es la variable objetivo (target)

### Clasificación binaria

En este proyecto, la tarea es una **clasificación binaria**:
- **Clase 0** (negativa): El estudiante alcanzará ≥ 70 puntos en la calificación final
- **Clase 1** (positiva): El estudiante **NO** alcanzará 70 puntos → riesgo de no promoción

### Modelos candidatos evaluados

| Modelo | Características | Ventaja |
|--------|----------------|---------|
| **Regresión Logística** | Lineal, probabilístico, interpretable | Coeficientes explicables, calibración natural |
| **Árbol de Decisión** | No lineal, reglas if-then | Visualización directa de las reglas |
| **Random Forest** | Ensamble de árboles, bagging | Robustez ante sobreajuste |
| **Gradient Boosting (Hist)** | Ensamble secuencial, boosting | Alto rendimiento en datos tabulares |

### Métricas clave

- **Recall (Sensibilidad):** Proporción de estudiantes en riesgo real que el modelo detecta. Es la métrica prioritaria porque un falso negativo (estudiante que reprueba sin alerta) tiene un costo 5x mayor que una falsa alarma.
- **Precisión:** Proporción de alertas que corresponden a riesgo real.
- **F1:** Media armónica entre precisión y recall.
- **F2:** Variante de F-score que pondera más el recall (usada para optimización).
- **ROC-AUC:** Capacidad discriminativa global del modelo.

### Umbral de decisión asimétrico

El umbral no es el 0.5 por defecto. Se eligió modelando:
- **Costo asimétrico 5:1** — Un falso negativo cuesta 5 veces más que un falso positivo
- **Restricción operativa** — No alertar a más del 35% del aula

Resultado: **Umbral = 0.37** con bandas:
- **BAJO:** probabilidad < 0.22
- **MEDIO:** 0.22 ≤ probabilidad < 0.37
- **ALTO:** probabilidad ≥ 0.37

---

## 3. Sistema Educativo Dominicano (MINERD)

### Estructura de evaluación del Nivel Secundario

| Elemento | Valor |
|----------|-------|
| Escala de calificación | 0 – 100 puntos |
| Nota mínima de promoción | 70 puntos |
| Períodos académicos | 4 (P1, P2, P3, P4) |
| Competencias fundamentales | 4 (C1-C4) |
| Calificación Final (C.F.) | Promedio de los 4 períodos |

### Competencias fundamentales del currículo

1. **C1 — Comunicativa:** Comprensión y producción de textos
2. **C2 — Pensamiento Lógico, Crítico y Creativo:** Resolución de problemas
3. **C3 — Ética y Ciudadana:** Desarrollo personal y espiritual
4. **C4 — Científica y Tecnológica:** Ambiental y de la salud

### Vías de promoción

```
Si C.F. ≥ 70  →  APROBADO (vía ordinaria)
Si C.F. < 70  →  Prueba Completiva
    C.C.F. = 50% × C.F. + 50% × Prueba Completiva
    Si C.C.F. ≥ 70  →  APROBADO (vía completiva)
    Si C.C.F. < 70  →  Prueba Extraordinaria
        C.EX.F. = 30% × C.F. + 70% × Prueba Extraordinaria
        Si C.EX.F. ≥ 70  →  APROBADO (vía extraordinaria)
        Si C.EX.F. < 70  →  NO PROMOVIDO
```

### Libro de registro escolar

El libro del docente (`3RO_B_REGISTRO_ESCOLAR_2026_2027.xlsx`) contiene:
- **Hoja L:** Listado de matrícula (nombres, cursos, secciones)
- **Hoja C:** Catálogo curricular (asignaturas × competencias)
- **Hoja Calificación:** Notas por período y calificación final
- **Hoja Prácticas:** Evaluación por competencia
- **Hoja Asistencia:** Registro diario de asistencia

---

## 4. Ética en Machine Learning Educativo

### Principios adoptados

1. **No discriminación:** Variables como nombre, sexo y nacionalidad se excluyen del modelo.
2. **Transparencia:** El modelo es explicable (regresión logística con coeficientes interpretables).
3. **Autonomía del docente:** El sistema es apoyo, nunca veredicto automático.
4. **Privacidad:** La landing page no envía datos a ningún servidor externo.
5. **Evitar profecía autocumplida:** La salida es un plan de acompañamiento, no una predicción de fracaso.

### Auditoría de equidad

Se verificó que las métricas del modelo no varían significativamente según:
- Grado (4to, 5to, 6to)
- Tanda (Matutina, Vespertina, Extendida)
- Nivel de apoyo familiar (Bajo, Medio, Alto)
