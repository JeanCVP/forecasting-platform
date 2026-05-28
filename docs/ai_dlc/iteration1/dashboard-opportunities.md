# Dashboard Opportunities
**AI-DLC — Initial Assessment | Phase 0**
**Generated:** 2026-05-21

---

## 1. Dashboard Portfolio Overview

| Dashboard | Audience | Refresh | Priority |
|---|---|---|---|
| Executive Supply Chain Command Center | C-Suite / Commercial Director | Weekly | P0 |
| Demand Planning Workbench | Demand Planners | Weekly | P0 |
| Channel Health Monitor | Sales Managers | Weekly | P1 |
| SKU Performance Tracker | Category Managers | Weekly | P1 |
| Forecast Accuracy Dashboard | ML / Analytics Team | Weekly | P1 |
| Inventory Risk Radar | Supply Chain Managers | Weekly | P2 |

---

## 2. Dashboard 1: Executive Supply Chain Command Center

**Audience:** Commercial Director, VP Supply Chain, CEO
**Purpose:** Single-page executive view of supply chain health and demand performance

### KPI Cards (Top Row)
| KPI | Formula | Format | Alert |
|---|---|---|---|
| Total Sell-in YTD | SUM(Sell-in, year=current) | Units / ▲% YoY | If <−5% vs prior year |
| Total POS YTD | SUM(Cust. Sales, year=current) | Units / ▲% YoY | If <−10% vs prior year |
| Avg. Channel DOS | Mean(Days of Supply) across active series | Days | If >60 or <14 days |
| Sell-through Rate | Cust. Sales / Sell-in (YTD) | % | If <70% |
| Active SKUs | Count(SKUs with Sales > 0 last 4w) | # | — |
| Active Channels | Count(Channels with Sales > 0 last 4w) | # | — |

### Charts
1. **Weekly Sell-in + Sell-out Trend (Line Chart)**
   - X: Week (W01 to current)
   - Y: Units (weekly)
   - Series: Sell-in (blue), Cust. Sales (orange), Channel Inv. (grey, right axis)
   - Annotations: key promotional events

2. **YoY Comparison Bar Chart**
   - X: Week of year
   - Y: Sell-in (current year vs. prior year)
   - Grouped bars; highlight weeks with >20% deviation

3. **Channel DOS Heatmap**
   - Rows: Top 20 channels by volume
   - Columns: Product families
   - Color: DOS (<14d = red, 14–30d = green, 30–60d = yellow, >60d = orange)

4. **Portfolio Performance Matrix (Scatter)**
   - X: Sell-through rate (%)
   - Y: Sell-in volume (units)
   - Color: Product family
   - Size: Channel inventory
   - Quadrant labels: Stars / Cash Cows / Slow Movers / Dead Stock

### Filters
- Year / Period selector
- Channel multi-select
- Product family multi-select
- Week range slider

---

## 3. Dashboard 2: Demand Planning Workbench

**Audience:** Demand Planners, S&OP Analysts
**Purpose:** Operational tool for reviewing, adjusting, and approving forecasts

### KPI Cards
| KPI | Description |
|---|---|
| Forecast Accuracy (MAPE) | Mean Absolute Percentage Error vs. actuals (last 8 weeks) |
| Bias | Mean signed error (over/under forecasting tendency) |
| Forecast Coverage | % of SKU-Channel pairs with valid forecast |
| SKUs needing review | Count of series with MAPE >30% |

### Charts
1. **Forecast vs. Actual Line (per SKU-Channel selection)**
   - Shows historical actuals + forecast with confidence bands
   - Editable override layer

2. **Error Distribution Histogram**
   - X: Forecast error %
   - Y: Count of series
   - Reference line at 0

3. **Accuracy by Product Family (Grouped Bar)**
   - X: Product family
   - Y: MAPE %
   - Color: Series segment (Regular / Intermittent / Rare)

4. **Forecast Adjustment Log (Table)**
   - Shows manual overrides with reason codes
   - Delta: System forecast vs. adjusted forecast

### Interactive Features
- SKU-Channel searchable selector
- Manual forecast override input (with reason code)
- Bulk upload of external adjustments
- Approval workflow toggle

### Filters
- Product family
- Channel
- Forecast week range
- Series segment (Regular / Intermittent / Rare)
- Error threshold (show only high-error series)

---

## 4. Dashboard 3: Channel Health Monitor

**Audience:** Sales Managers, Key Account Managers
**Purpose:** Per-channel inventory health and sell-through performance

### KPI Cards (per selected channel)
| KPI | Formula |
|---|---|
| Channel Total Sell-in (W) | SUM Sell-in current week |
| Channel Total POS (W) | SUM Cust. Sales current week |
| Channel Avg. DOS | Mean Days of Supply across active SKUs |
| SKUs at Risk (DOS>60d) | Count of overstocked SKUs |
| SKUs Stocked Out (DOS<7d) | Count of near-stockout SKUs |

### Charts
1. **Inventory Health Gauge (Radial)**
   - Shows % of channel's SKUs in healthy DOS zone (14–45 days)
   - Color zones: Critical / Healthy / Overstocked

2. **SKU Inventory Ranking (Horizontal Bar)**
   - Top 20 SKUs by Days of Supply
   - Color coded by risk zone
   - Secondary bar: weekly POS velocity

3. **Channel Sell-in vs. Sales Trend (Area Chart)**
   - Weekly Sell-in vs. Cust. Sales
   - Shaded area = inventory gap (positive = building, negative = depleting)

4. **SKU Sell-through Table**
   - Columns: SKU | Sell-in (4w) | Sales (4w) | Sell-through % | DOS | Trend
   - Sortable; exportable to Excel

### Filters
- Channel selector (single or compare 2)
- Product family
- Date range
- DOS threshold alert setting

---

## 5. Dashboard 4: SKU Performance Tracker

**Audience:** Category Managers, Product Managers
**Purpose:** Product lifecycle, performance, and distribution breadth

### KPI Cards (per product family)
| KPI | Description |
|---|---|
| Active SKU Count | SKUs with sales in last 4 weeks |
| New SKUs (last 8w) | Product introductions |
| Retiring SKUs | No sales in last 8 weeks |
| Avg. Channel Reach | Mean channels per active SKU |
| Top SKU Sell-through | Best performing SKU's rate |

### Charts
1. **SKU Lifecycle Gantt Chart**
   - Y: SKU (sorted by launch date)
   - X: Weeks
   - Bar: Active period (first to last non-zero sale)
   - Color: Product family

2. **BCG-style Portfolio Matrix**
   - X: Sales growth rate (YoY)
   - Y: Channel reach (number of channels)
   - Bubble size: Total volume
   - Quadrants: Stars / Question Marks / Cash Cows / Dogs

3. **SKU Distribution Heatmap**
   - Rows: SKUs (top 50 by volume)
   - Columns: Channels
   - Color: Sales volume (log scale)
   - Reveals which SKUs have broad vs. narrow distribution

4. **New SKU Adoption Curve**
   - X: Weeks since launch
   - Y: Cumulative sales / channels
   - Multiple SKUs overlaid for comparison

### Filters
- Product family
- Launch year/cohort
- Minimum sales threshold
- Channel subset

---

## 6. Dashboard 5: Forecast Accuracy Dashboard

**Audience:** Data Science / Analytics Team, Demand Planning Lead
**Purpose:** Model performance monitoring and diagnostics

### KPI Cards
| KPI | Description |
|---|---|
| MAPE (Overall) | Across all series, all horizons |
| MAPE (H+1) | Next-week accuracy |
| MAPE (H+4) | 4-week-ahead accuracy |
| MAPE (H+8) | 8-week-ahead accuracy |
| Bias (%) | Systematic over/under-forecasting |
| Coverage >20% error | % of series with unacceptable error |

### Charts
1. **Accuracy Over Time (Line)**
   - X: Forecast week (as of date)
   - Y: MAPE %
   - Series by model variant and horizon

2. **Error by Segment (Box Plot)**
   - X: Series segment (Regular / Intermittent / Rare)
   - Y: MAPE distribution
   - Compare model versions side by side

3. **Residual Plot**
   - X: Actual value
   - Y: Forecast error
   - Colored by product family
   - Should show no pattern if model is unbiased

4. **Forecast Horizon Decay Curve**
   - X: Forecast horizon (weeks ahead: H+1 to H+19)
   - Y: MAPE %
   - Shows how accuracy degrades with horizon length

5. **Top Worst Performers Table**
   - SKU-Channel pairs with highest MAPE
   - Drilldown to actual vs. forecast chart

---

## 7. Dashboard 6: Inventory Risk Radar

**Audience:** Supply Chain Managers, Logistics
**Purpose:** Early warning system for overstock and stockout risks

### KPI Cards
| KPI | Alert Threshold |
|---|---|
| SKUs at Overstock Risk | DOS > 60 days |
| SKUs at Stockout Risk | DOS < 7 days |
| Total Excess Inventory Units | Inv above 60-day DOS target |
| % Channel Inventory Healthy | In 14–45 day DOS range |

### Charts
1. **Risk Matrix (Scatter)**
   - X: Days of Supply
   - Y: Weekly Sales Velocity (units/week)
   - Color: Product family
   - Vertical lines at 7d and 60d thresholds
   - Size: Total inventory value (if available)

2. **Risk by Channel (Stacked Bar)**
   - X: Channel
   - Y: SKU count
   - Stack: Stockout risk / Healthy / Overstock risk
   - Sorted by risk count descending

3. **Inventory Trend Alerts (Table)**
   - Series trending toward stockout (DOS declining)
   - DOS current | DOS 4 weeks ago | Trend arrow | Days to stockout (at current rate)

4. **Geographic/Channel Map** (if channel geo available)
   - Map of Colombia with channel locations
   - Color: Inventory health
   - Click for channel detail

### Filters
- Risk type (overstock / stockout / all)
- Product family
- Channel
- DOS threshold sliders

---

## 8. Technical Implementation Notes

| Component | Recommendation |
|---|---|
| **BI Tool** | Power BI, Looker, or Tableau (all compatible with weekly time-series format) |
| **Data refresh** | Weekly (after data ingestion pipeline runs, typically Monday AM) |
| **Data model** | Star schema: Fact table (weekly observations) + Dim tables (Channel, Material, Calendar, Forecast) |
| **Calendar dimension** | Include ISO week, month, quarter, holidays, Colombian events |
| **Forecast table** | Store forecast alongside actuals with model_version, forecast_date, horizon |
| **Performance** | Aggregate to channel × product family for executive dashboards; drill to SKU level on demand |
| **Export** | All tables exportable to Excel for demand planners |

---

## 9. KPI Reference Card

| KPI | Formula | Good | Warning | Critical |
|---|---|---|---|---|
| Days of Supply | Inv / (Sales/7) | 14–45 days | 7–14 or 45–60 | <7 or >60 |
| Sell-through Rate | Sales / Sell-in | >80% | 60–80% | <60% |
| Forecast MAPE | |Actual−Fcst|/Actual | <15% | 15–30% | >30% |
| Forecast Bias | Mean(Fcst−Actual)/Actual | ±5% | ±5–15% | >±15% |
| Inventory Coverage | Inv / avg 4w Sales | 2–6 weeks | 1–2 or 6–10 | <1 or >10 |
| Channel Reach | # channels active for SKU | >10 | 3–10 | <3 |
| Portfolio Sell-through | Total Sales / Total Sell-in YTD | >75% | 60–75% | <60% |

---

*Document Version: 1.0 | AI-DLC Traceability ID: ASSESSMENT-2026-001-DO*
