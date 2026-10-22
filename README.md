# Global Tuberculosis (TB) Epidemiological Profiling & Predictive Modeling

This repository contains an exploratory data analysis (EDA) and predictive machine learning framework designed to profile public health outcomes and risk vectors associated with Tuberculosis (TB). Operating over a structured cohort of **400,000 comprehensive records**, this project investigates how demographic traits, healthcare infrastructure, and socio-economic contexts influence global clinical distributions.

## 📌 Project Overview

Tuberculosis remains a critical global health concern. This repository establishes a structural pipeline to ingest, clean, and profile extensive health registries. The analytical architecture is engineered to run seamlessly within an isolated virtual environment shell (`.venv`) utilizing modern scientific Python modules.

### Core Metrics Covered
*   **Epidemiological Risk Vectors:** TB Incidence, Mortality Rates, HIV Co-Infection Rates, and Multidrug Resistance (MDR).
*   **Socio-Demographic Stratification:** Age Brackets, Gender Groups, Urban vs. Rural Population Percentages, and Regional Unemployment Rates.
*   **Socio-Economic & Infrastructural Health Indicators:** GDP Per Capita, Healthcare Expenditure Per Capita, BCG Vaccination Coverage, and Primary Healthcare Access.

---

## 🛠️ Repository & Environment Configuration

The codebase is optimized for **Python 3.10+** setups and strictly isolates dependencies within a dedicated local runtime environment shell (`.venv`).

### Architectural Pipeline Prerequisites
*   `scikit-learn` (v1.9.1+) — Statistical evaluation and predictive model execution.
*   `pandas` — Vectorized tabular data manipulations.
*   `numpy` (v2.5.4+) — Optimized numerical array structures.
*   `scipy` & `joblib` — Scientific computation backends and pipeline caching.
*   `matplotlib` & `seaborn` — Descriptive distribution rendering and statistical plotting.

### Setup and Replication Steps
Initialize your virtual environment terminal wrapper from the project workspace to unpack required distributions cleanly:

```bash
# Navigate to project root directory
cd "intro to ds assn 1"

# Initialize local virtual environment wrapper
python -m venv .venv

# Activate local environment runner paths (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Refresh structural packaging pipelines and dependencies
pip install --upgrade pip
pip install scikit-learn pandas numpy matplotlib seaborn
```

---

## 📊 Exploratory Data Analysis & Structural Profiling

### 1. Ingestion Profiling
The analytical entry point maps data sources directly via string literals to guarantee robust path resolution across Windows execution threads:
```python
import pandas as pd

# Structural data load sequence 
tb_data = pd.read_csv(r'C:\Users\Hp\Desktop\INTRO TO DS ASSN 1\Tuberculosis_Dataset.csv')
print(f"Dataset successfully loaded! Dimensions: {tb_data.shape}")
# Outputs: Dataset successfully loaded! Dimensions: (400000, 22)
```

### 2. Feature Architecture
The pipeline maps **22 explicit features** consisting of 17 float-point distributions, 2 integer features, and 3 categorical string layers:

| Feature Name | Storage Type | Domain Context |
| :--- | :--- | :--- |
| `Country` / `Year` | `str` / `int64` | Spatiotemporal identifiers |
| `TB_Incidence_Rate` | `float64` | Disease occurrence metric (Target Vector) |
| `HIV_Prevalence` / `HIV_CoInfection_Rate` | `float64` | Retroviral baseline and direct clinical overlap |
| `BCG_Vaccination_Coverage` | `float64` | Preventative care distribution density |
| `Multidrug_Resistance_Rate` | `float64` | Antimicrobial resistance baseline |
| `Healthcare_Expenditure_Per_Capita` | `float64` | Direct economic infrastructural backing |
| `Age_Group` / `Gender` | `str` | Demographic grouping features |

### 3. Distribution Metrics (Averages Overview)
Statistical descriptions are evaluated using transposed frames (`.describe().T`) to handle wide dimensional structures cleanly. Sample distribution medians match means closely, pointing to stable, well-scaled dataset vectors:
*   **Target Baseline (TB Incidence):** Mean ~`150.01` | Median ~`149.99` | Max ~`383.10` cases per cohort block.
*   **Clinical Vulnerability (HIV Co-Infection):** Mean ~`2.50%` | Median ~`2.51%` | Max ~`5.00%`.
*   **Economic Context (GDP Per Capita):** Mean ~`$50,247.13` | Median ~`$50,275.20` | Max ~`$99,999.87`.

---

## 🔍 Data Validation Framework
The code features built-in verification scripts that calculate missing cellular allocations dynamically. This ensures proper data density before passing vectors into **scikit-learn** estimators:
```python
# Validation check: Percentage-based missing data report
missing_count = tb_data.isnull().sum()
missing_percentage = (tb_data.isnull().sum() / len(tb_data)) * 100
```
