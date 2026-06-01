# SLIDES — Hackathon SIC2025
## Análisis Predictivo de Demanda e Inventario · Samsung Colombia
**Guía para Canva / PowerPoint — Diseño minimalista y ejecutivo**

---

## SLIDE 1 — PORTADA

**Título:** Análisis Predictivo de Demanda e Inventario  
**Subtítulo:** Hackathon SIC2025 · Samsung Colombia  
**Datos clave (badges):**
- 3 años de datos históricos (2023–2025)
- 98 clientes · 2,093 productos
- Frecuencia semanal

*Visual: fondo oscuro, logo Samsung, ícono de gráfico de pronóstico*

---

## SLIDE 2 — EL PROBLEMA

**Título:** ¿Por qué necesitamos análisis predictivo?

**3 bloques:**
1. **Sin análisis predictivo:** Decisiones basadas en intuición → quiebres de stock, clientes perdidos, inventario excesivo
2. **Con nuestro modelo:** Anticipar demanda con 3.4× más precisión que el método actual
3. **Impacto directo:** Menos quiebres · Más retención de clientes · Inventario optimizado

*Visual: flecha de izquierda (caos) a derecha (orden)*

---

## SLIDE 3 — LOS DATOS

**Título:** Estructura de los datos

**Tabla visual:**
| Dimensión | Valor |
|---|---|
| Período | 2023 – W33·2025 (134 semanas) |
| Clientes (Channels) | 98 |
| Productos (SKUs) | 2,093 |
| Registros | 4.6 millones |
| Granularidad | Semanal por SKU |

**3 categorías clave:**
- 🔵 **Sell-in** — Ventas al canal distribuidor
- 🟢 **Cust. Sales** — Ventas al consumidor final
- 🟡 **Channel Inv.** — Inventario disponible

*Visual: ícono de base de datos, tabla simple*

---

## SLIDE 4 — HALLAZGO CLAVE: INTERMITENCIA

**Título:** La demanda es altamente intermitente

**Número grande:** `99%`  
**Subtexto:** de las semanas no tienen venta para la mayoría de SKUs

**Lo que esto significa:**
- Los métodos estadísticos convencionales fallan
- Se necesita un modelo especializado para demanda intermitente
- El clasificador "¿habrá demanda?" es tan importante como el regresor "¿cuánto?"

*Visual: gráfico de barras mostrando mayoría de ceros*

---

## SLIDE 5 — ESTACIONALIDAD Y PATRONES

**Título:** ¿Cuándo se vende más?

**Hallazgos estadísticos:**
- 📈 **Mejor mes histórico:** Diciembre 2024 — 591K unidades
- 🏆 **Semanas pico:** S22, S26, S17 (mayo-junio y agosto)
- 📊 **Tendencia:** Crecimiento 2023→2024, corrección en 2025
- 🗓️ **Patrón:** La demanda se concentra en mitad del año

**Recomendación:** Preparar inventario en semanas 15–20 y 30–35

*Visual: gráfico semanal con semanas peak marcadas en rojo*

---

## SLIDE 6 — EL MODELO PREDICTIVO

**Título:** Arquitectura del modelo LightGBM Two-Stage

**Flujo visual (3 cajas):**

```
[ETAPA 1 — CLASIFICADOR]          [ETAPA 2 — REGRESOR]         [PREDICCIÓN FINAL]
¿Habrá demanda esta semana?   →   ¿Cuánto se venderá?      →   Pronóstico con
LightGBM binario                  LightGBM Huber + log-         bandas Q10/Q90
AUC = 0.964                       transform del target
                                  Optimizado con Optuna
```

**21 features:** Lags 1/4/52 semanas · Medias móviles · Estacionalidad · Zscore vs canal

*Visual: diagrama de flujo limpio en 3 pasos*

---

## SLIDE 7 — VALIDACIÓN DEL MODELO

**Título:** El modelo supera al baseline en todos los escenarios

**Tabla de resultados CV (5 folds walk-forward):**

| Métrica | Seasonal Naïve (base) | Nuestro Modelo | Mejora |
|---|---|---|---|
| **MASE** | 1.659 | **0.495** | **3.4× mejor** |
| **demand_F1** | — | **0.775** | Acierta 77% |
| **AUC** | — | **0.964** | Clasif. excelente |

**Nota metodológica:** Validación walk-forward = sin fuga de datos. El modelo entrena solo con datos pasados y predice datos futuros no vistos.

*Visual: barra comparativa Naïve vs Modelo, flecha de mejora*

---

## SLIDE 8 — PREGUNTA 1: Mayor Rotación

**Título:** ¿Qué productos liderarán la demanda en 2026?

**Top 5 (barras horizontales):**
1. MOBILE SM-A165M/DS Black — 106K unidades
2. MOBILE SM-A165M/DS Gray — 76K unidades
3. MOBILE SM-A055M/DS Black — 59K unidades
4. MOBILE SM-A065M/DS Black — 58K unidades
5. MOBILE SM-A165M/DS Light Green — 56K unidades

**Insight:** La categoría MOBILE domina el pronóstico. SM-A165M en sus 3 variantes de color suma ~237K unidades estimadas para 2026.

*Visual: gráfico de barras horizontales en azul*

---

## SLIDE 9 — PREGUNTA 2: Riesgo de Churn

**Título:** ¿Qué clientes podrían abandonar?

**Distribución (pie chart):**
- 🔴 **ALTO** (>26 sem sin comprar): 16 clientes — **19%**
- 🟠 **MEDIO** (13–26 sem o tendencia bajista): 9 clientes — **11%**
- 🟢 **ACTIVO** (comportamiento estable): 58 clientes — **70%**

**Acción recomendada:**
- Los 16 clientes ALTO: visita comercial urgente, propuesta de reactivación
- Los 9 clientes MEDIO: monitoreo mensual, alerta temprana
- Oportunidad: retener 30% de clientes en riesgo podría recuperar X unidades de venta

*Visual: donut chart con colores semáforo*

---

## SLIDE 10 — PREGUNTA 3: Crecimiento y Caída

**Título:** ¿Qué productos crecerán y cuáles caerán?

**Dos columnas:**

| 📈 Productos que CRECEN (H2 vs H1 2026) | 📉 Productos en DECLIVE |
|---|---|
| AV Receiver HW-B400F (+188K%) | Productos con lag_52=0 sin señal estacional |
| MOBILE SM-A165M variantes | SKUs de canales en riesgo churn |
| Monitor S19A330NHL | TVs de gama baja sin demanda reciente |

**Insight:** El crecimiento se concentra en audio (AV Receiver) y smartphones de gama media. Los productos en declive están ligados a los clientes en riesgo de churn.

---

## SLIDE 11 — PREGUNTA 4: Despacho Urgente

**Título:** ¿A quién despachar más producto y cuál?

**Semáforo de riesgo:**
- 🔴 **171 SKUs CRITICAL** — menos de 2 semanas de cobertura
- 🟠 **33 SKUs HIGH** — 2 a 4 semanas de cobertura

**Cliente prioritario: CUSTOMER35**
- Múltiples TVs (QLED 50", LED 55"/75") con inventario < 1 unidad
- Quiebre proyectado: W01–W07·2026
- Acción: Despacho inmediato de QLED QN50QEF1AK, LED UN75DU8000K y Monitor S32DM801UN

*Visual: tabla de los 5 SKUs más urgentes con semáforo de color*

---

## SLIDE 12 — DASHBOARD INTERACTIVO

**Título:** Todos los resultados en un dashboard en tiempo real

**5 páginas disponibles:**
- 🏠 Resumen Ejecutivo — KPIs globales y alertas principales
- 📈 Comportamiento de Ventas — histórico 2023–2025
- 🔮 Proyección de Demanda — forecast 52 semanas (W34·2025–W33·2026) con banda Q10/Q90
- 🚨 Alertas de Inventario — riesgo CRITICAL/HIGH/MEDIUM/LOW por SKU
- 👥 Análisis de Churn — riesgo de abandono por cliente

**Tecnología:** Python · LightGBM · Streamlit · MLflow

---

## SLIDE 13 — CONCLUSIONES

**Título:** Resumen ejecutivo

**4 acciones inmediatas:**

| # | Acción | Impacto |
|---|---|---|
| 1 | Despachar a CUSTOMER35 (TVs y monitores) | Evitar quiebre W01·2026 |
| 2 | Plan de retención para 16 clientes ALTO churn | Recuperar relación comercial |
| 3 | Preparar inventario SM-A165M para H1·2026 | Capitalizar tendencia de crecimiento |
| 4 | Alertas semanales automáticas con el modelo | Decisiones proactivas, no reactivas |

**El modelo predice con 3.4× más precisión que el método actual.**  
**Con análisis predictivo, el negocio pasa de reaccionar a anticipar.**

---

## NOTAS DE DISEÑO PARA CANVA

- **Fondo:** Blanco o gris muy claro (#F8F9FA)
- **Color principal:** Azul Samsung (#1428A0) para títulos y acentos
- **Secundario:** Verde (#0e9f6e) para positivo, Rojo (#e02424) para alertas
- **Tipografía:** SamsungOne o Inter (Google Fonts)
- **Gráficos:** Importar desde el notebook (exportar como PNG)
- **Slides:** 16:9, máximo 3 puntos por slide
- **Estilo:** Minimalista, sin fondo oscuro en body, datos prominentes
