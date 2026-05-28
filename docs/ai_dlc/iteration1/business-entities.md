# Business Entities & Domain Understanding
**AI-DLC — Initial Assessment | Phase 0**
**Generated:** 2026-05-21

---

## 1. Industry & Business Context

**Inferred Industry:** Consumer Electronics Manufacturing & Distribution
**Inferred Market:** Colombia (based on `COLOMBIA` suffix in ~100% of material descriptions)
**Business Model:** B2B2C — manufacturer sells to retail/distribution channels (Sell-in), who sell to end consumers (Cust. Sales)
**Product Families Observed:** Mobile phones, TVs (LED, QLED, OLED), Tablets, Monitors, AV Receivers, Refrigerators, Microwaves, Air Conditioners, Signage, Commercial displays

This profile is consistent with a major global consumer electronics brand operating its Colombia subsidiary or distribution entity, managing a network of 68–92 retail and distribution partners.

---

## 2. Core Business Entities

### Entity 1: Channel (Customer / Trade Partner)
| Attribute | Value |
|---|---|
| **Definition** | A retail chain, distributor, or commercial partner that purchases products from the manufacturer |
| **Identifier** | `CUSTOMER{N}` (anonymized) |
| **Cardinality** | 68 stable, up to 92 at peak |
| **Lifecycle** | Some channels appear/disappear year-over-year (churn ~20%/year) |
| **Hierarchy** | Not explicit in data — likely has sub-hierarchy (channel type: chain / independent / online / B2B) |
| **Role in Supply Chain** | Intermediary between manufacturer and end consumer |

### Entity 2: Material (SKU / Product)
| Attribute | Value |
|---|---|
| **Definition** | A specific sellable product variant (model + color + size + market configuration) |
| **Identifier** | `Material Description` (composite string) |
| **Cardinality** | 726 stable, up to 1,645 at peak |
| **Lifecycle** | High portfolio churn: ~42% of SKUs replaced annually |
| **Hierarchy** | Parseable from description: `Product Family > Model Code > Size/Variant > Color > Config` |
| **Geographic Scope** | All identified as COLOMBIA market variants |

### Entity 3: Metric Category (KPI Type)
| Attribute | Value |
|---|---|
| **Definition** | The business measurement type associated with a row |
| **Values** | `Sell-in`, `Cust. Sales`, `Channel Inv.` |
| **Semantics** | Flow (Sell-in, Sales) vs. Stock (Inventory) |

### Entity 4: Week (Time Period)
| Attribute | Value |
|---|---|
| **Definition** | ISO calendar week within a given year |
| **Format** | `{YYYY}{WW}` |
| **Coverage** | W01 2023 through W33 2025 (historical) |
| **Forecast Horizon** | W34–W52 2025 (target) |

---

## 3. Entity Relationships

```
CHANNEL (1) ────────< (M) CHANNEL_MATERIAL_SERIES
MATERIAL (1) ───────< (M) CHANNEL_MATERIAL_SERIES
CATEGORY (1) ───────< (M) CHANNEL_MATERIAL_SERIES

CHANNEL_MATERIAL_SERIES (1) ──< (52) WEEKLY_OBSERVATION
```

The central entity is the **Channel × Material × Category series** — a unique weekly time series of one metric for one product at one customer. There are approximately:

- 8,413 × 3 = **25,239 active series** in 2025 (before dedup aggregation)
- ~8,413 unique SKU-Channel pairs with inventory, sell-in, and sales streams each

---

## 4. Supply Chain Flow Model

```
MANUFACTURER
     │
     │  Sell-in (units shipped to channel)
     ▼
CHANNEL WAREHOUSE
     │  Channel Inv. (stock on hand at channel)
     │
     │  Cust. Sales (units sold to end consumers)
     ▼
END CONSUMER
```

**Key Inventory Identity:**
```
Channel Inv.(W) = Channel Inv.(W-1) + Sell-in(W) − Cust. Sales(W) + Adjustments
```

Deviations from this identity (returns, write-offs, stock corrections) represent an implicit **Adjustment** entity not present in the raw data.

---

## 5. Inferred Business Processes

### 5.1 Sell-in Planning (Demand Planning / S&OP)
The manufacturer plans how many units to ship to each channel each week. This is the primary **forecast target** — the company needs to predict Sell-in to optimize production and logistics.

### 5.2 Sell-out Monitoring (POS Intelligence)
Weekly POS (Point of Sale) data from channels enables sell-out tracking. The difference between Sell-in and Sell-out determines inventory build-up.

### 5.3 Inventory Management (Channel Stock Health)
Channel Inventory is monitored to avoid:
- **Overstock:** High carrying costs, markdown risk, liquidation
- **Stockout:** Lost sales, customer dissatisfaction, brand damage

**Key Derived KPI:** Days of Supply (DOS) = `Channel Inv. / (Cust. Sales / 7)`

### 5.4 Product Lifecycle Management
The high SKU churn rate (42% per year) reflects an active product lifecycle with annual model refreshes (common in consumer electronics). New model introductions require cold-start demand planning.

---

## 6. Inferred Business KPIs

| KPI | Formula | Source Data |
|---|---|---|
| Total Sell-in (units) | SUM(Sell-in weeks) | Sell-in category |
| Total POS (units) | SUM(Cust. Sales weeks) | Cust. Sales category |
| Sell-through Rate | Cust. Sales / Sell-in | Both |
| Channel Inventory Level | Channel Inv. (snapshot) | Channel Inv. category |
| Days of Supply (DOS) | Inv / (Sales/7) | Both |
| Inventory Coverage (weeks) | Inv / avg weekly Sales | Both |
| Replenishment Rate | Sell-in / avg weekly Sales | Both |
| Active SKU Count | Count of non-zero series | All |
| Channel Reach | Count of channels per SKU | All |
| Demand Volatility (CV) | StdDev(Sales) / Mean(Sales) | Cust. Sales |
| Sell-in Forecast Accuracy | MAPE(actual vs. forecast) | Sell-in |

---

## 7. Business Seasonality Patterns (Inferred)

From weekly average Sell-in analysis:

**2023 Peak Weeks:** W04, W12, W13, W17, W43 → aligns with:
- W04: Post-New Year retail restock
- W12–W13: Pre-Easter demand
- W17: Mother's Day (Colombia: May)
- W43: Pre-Black Friday / Colombian retail events

**2024 Peak Weeks:** W22, W26, W30, W39, W48 → aligns with:
- W22: Mid-year promotional period
- W26: Half-year restock
- W39: Pre-season
- W48: Black Friday / Cyber Monday preparation

**Low-Activity Weeks:** Consistently W14, W18, W27 (week following major events = demand trough)

---

## 8. Product Family Hierarchy (Inferred)

```
Electronics Portfolio
├── Mobile Devices (35% of SKUs)
│   ├── Smartphones (SM- prefix models)
│   └── Feature Phones
├── Televisions (21% of SKUs)
│   ├── LED TV
│   ├── QLED TV
│   └── OLED SCREEN
├── Computing & Display (11% of SKUs)
│   ├── MON (Monitors)
│   ├── TABLET
│   └── LFD / SIGNAGE
├── Audio & Entertainment (4% of SKUs)
│   └── AV RECEIVER / MINI COMPONENT
└── Home Appliances (7% of SKUs)
    ├── RTF (Refrigerators)
    ├── MWO (Microwaves)
    ├── CAC / RAC / DVM (Air Conditioning)
    └── OWM (Washing Machines)
```

---

## 9. Stakeholder Personas (Inferred)

| Persona | Needs from Forecast |
|---|---|
| **Demand Planner** | Weekly Sell-in forecast by SKU-Channel; accuracy metrics |
| **Sales Manager** | Sell-through rates by channel; channel health dashboard |
| **Supply Chain Manager** | Inventory risk alerts; DOS by SKU-Channel |
| **Category Manager** | Product performance trends; portfolio rationalization signals |
| **Commercial Director** | Executive KPI dashboard; forecast vs. actual at brand level |

---

*Document Version: 1.0 | AI-DLC Traceability ID: ASSESSMENT-2026-001-BE*
