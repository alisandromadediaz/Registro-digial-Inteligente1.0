# 📘 Introducción

## Título del Proyecto

**Registro Digital Inteligente: Sistema de Alerta Temprana de No Promoción para el Nivel Secundario Dominicano**

**Autor:** Alisandro Made  
**Metodología:** CRISP-ML(Q) — Cross Industry Standard Process for Machine Learning with Quality Assurance  
**Año escolar:** 2026-2027  
**Institución:** Sistema Educativo Dominicano (MINERD)

---

## Planteamiento del Problema

En el Nivel Secundario del Sistema Educativo Dominicano, el docente registra de manera sistemática toda la información necesaria para el seguimiento académico del estudiante: calificaciones por período (P1-P4), prácticas por competencia (C1-C4) y asistencia diaria. Sin embargo, esta información **se consulta cuando ya es tarde**: el estudiante llega a la prueba completiva o extraordinaria sin que nadie haya intervenido a tiempo.

### Problemática específica

1. **Detección tardía del riesgo académico:** Las alertas se generan al final del año escolar, cuando las opciones de intervención son limitadas.
2. **Desaprovechamiento de datos existentes:** El libro de registro escolar contiene información valiosa que no se utiliza de forma predictiva.
3. **Ausencia de herramientas tecnológicas:** No existe un sistema automatizado que permita al docente anticipar qué estudiantes están en riesgo de no promoción.
4. **Falta de planes de acompañamiento personalizados:** Las intervenciones pedagógicas son genéricas y no se adaptan al perfil de riesgo individual.

### Pregunta de investigación

> ¿Es posible construir un modelo de Machine Learning supervisado que, utilizando únicamente los indicadores observables al cierre del Período 2, prediga con alta sensibilidad qué estudiantes no alcanzarán los 70 puntos en la calificación final ordinaria, para activar a tiempo la Recuperación Pedagógica?

---

## Justificación

Este proyecto se justifica desde múltiples perspectivas:

### Perspectiva educativa
- Permite la **intervención temprana** al identificar estudiantes en riesgo a mitad del año escolar.
- Transforma datos que ya se recopilan en **información accionable** para el docente.
- Genera **planes de acompañamiento concretos** vinculados a los indicadores del estudiante.

### Perspectiva tecnológica
- Demuestra la aplicabilidad del **Machine Learning supervisado** en el contexto educativo dominicano.
- Implementa la metodología **CRISP-ML(Q)** con aseguramiento de calidad en cada fase.
- Despliega una solución que funciona **sin conexión a internet** y **sin compartir datos del estudiante**.

### Perspectiva ética
- Excluye variables sensibles (nombre, sexo, nacionalidad) del entrenamiento.
- El resultado no es una etiqueta de fracaso, sino un **plan de acompañamiento**.
- La decisión siempre corresponde al **docente y al equipo de gestión**.

### Perspectiva metodológica
- Aplica el estándar industrial CRISP-ML(Q) completo (6 fases) en un proyecto educativo.
- Documenta cada decisión de diseño, desde la selección de variables hasta el umbral de decisión.
- Incluye auditoría de equidad y análisis de errores.

---

## Alcance del Proyecto

| Aspecto | Definición |
|---------|-----------|
| **Tipo de aprendizaje** | Supervisado |
| **Tarea** | Clasificación binaria |
| **Unidad de análisis** | Par *(estudiante, asignatura)* dentro de un año escolar |
| **Variable objetivo** | `riesgo_no_promocion` = 1 si C.F. ordinaria < 70 |
| **Momento de predicción** | Cierre del Período 2 (mitad del año) |
| **Población** | 195 estudiantes del Nivel Secundario (3 años escolares simulados) |
| **Registros** | 3,510 pares (estudiante, asignatura) |

---

## Fuente de Datos

El proyecto utiliza como fuente primaria el libro de registro escolar del docente:

📁 `3RO_B_REGISTRO_ESCOLAR_2026_2027.xlsx`

Este libro contiene la matrícula real del centro educativo con 195 estudiantes distribuidos en 6 cursos, 20 asignaturas del catálogo curricular y las reglas de promoción del MINERD.

> **Nota importante:** Dado que el año escolar 2026-2027 apenas inicia, las celdas de calificación están vacías. El histórico de desempeño fue simulado calibrándolo sobre la estructura real del centro, como se documenta en la sección de Metodología.
