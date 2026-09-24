# ✅ Conclusiones y Recomendaciones

## Conclusiones

### 1. Viabilidad del enfoque
Se demostró que es posible construir un modelo de Machine Learning supervisado que, utilizando únicamente los indicadores observables al cierre del Período 2 del registro escolar del docente, prediga con alta sensibilidad (Recall = 0.946) qué estudiantes están en riesgo de no alcanzar los 70 puntos en la calificación final ordinaria.

### 2. Cumplimiento de criterios de éxito
Los cuatro criterios de éxito definidos en la Fase 1 de CRISP-ML(Q) se cumplieron satisfactoriamente:

| Criterio | Meta | Logro |
|----------|------|-------|
| Recall ≥ 0.90 | Detectar 9 de 10 en riesgo | 0.946 ✅ |
| Precisión ≥ 0.80 | ≤ 2 falsas alarmas por 10 | 0.859 ✅ |
| % alertado ≤ 35% | Carga operativa viable | 34.6% ✅ |
| ROC-AUC ≥ 0.95 | Discriminación alta | 0.987 ✅ |

### 3. Interpretabilidad
La elección de Regresión Logística como modelo final permite que el docente comprenda *por qué* un estudiante recibe una alerta, lo cual es esencial para generar confianza en el sistema y para diseñar intervenciones pedagógicas específicas.

### 4. Despliegue accesible
El sistema se desplegó en tres formatos complementarios:
- **Aplicación Streamlit** para uso diario en el centro educativo.
- **Landing page HTML autónoma** que funciona sin internet ni servidor, protegiendo la privacidad del estudiante.
- **Cuaderno Jupyter** reproducible que documenta todo el proceso con metodología CRISP-ML(Q).

### 5. Diseño ético
Se implementaron salvaguardas éticas desde el diseño:
- Variables sensibles excluidas del modelo.
- Auditoría de equidad por subgrupos.
- El resultado es un plan de acompañamiento, no una etiqueta de fracaso.
- La decisión siempre corresponde al docente.

### 6. Robustez técnica
El sistema es robusto ante errores de tipo (`TypeError`), con funciones defensivas que normalizan todas las entradas provenientes de Excel, formularios HTML o CSV. El cuaderno completo (60 celdas) ejecuta sin errores.

---

## Recomendaciones

### Para la implementación inmediata

1. **Validación con datos reales:** Al cierre del Período 2 del año 2026-2027, sustituir la simulación por las calificaciones reales del libro de registro. El contrato de datos en `src/registro_ia.py` garantiza que no se requieren cambios en el resto del sistema.

2. **Capacitación docente:** Formar al equipo docente en el uso de la herramienta, enfatizando que es un apoyo y no un veredicto automático.

3. **Piloto controlado:** Implementar el sistema en un curso piloto antes de desplegarlo en todo el centro educativo, para medir el impacto real en la tasa de promoción.

### Para el desarrollo futuro

4. **Explicaciones locales (SHAP):** Incorporar explicaciones SHAP para justificar cada alerta individual con los factores específicos que más contribuyen al riesgo.

5. **Aprendizaje continuo:** Incorporar los registros generados por la aplicación al conjunto de entrenamiento para mejorar el modelo con cada ciclo escolar.

6. **Integración con SGCE:** Conectar el sistema con el Sistema de Gestión de Centros Educativos del MINERD para automatizar la entrada de datos.

7. **Evaluación de impacto:** Diseñar un estudio cuasi-experimental que compare la tasa de promoción de cursos con y sin el sistema durante un año escolar completo.

8. **Extensión a otros niveles:** Adaptar el modelo para el Nivel Primario y para la educación técnico-profesional, ajustando las variables y las reglas de promoción.

---

## Limitaciones Reconocidas

1. **Datos simulados:** El modelo actual fue entrenado con desempeño simulado. Las métricas no son evidencia sobre estudiantes reales.

2. **Eventos imprevistos:** No predice factores posteriores al Período 2 (enfermedad, migración, trabajo).

3. **Contexto específico:** El modelo fue calibrado sobre un centro educativo específico y puede requerir ajustes para otros contextos.

4. **Dependencia de la calidad del registro:** La calidad de las predicciones depende de la fidelidad con que el docente registra los indicadores en el libro.

---

## Reflexión Final

> El Registro Digital Inteligente demuestra que los datos que el docente ya recopila contienen información predictiva valiosa. La tecnología de Machine Learning, aplicada con rigor metodológico y responsabilidad ética, puede transformar esos datos en intervenciones oportunas que mejoren las trayectorias educativas de los estudiantes dominicanos.

**— Alisandro Made**
