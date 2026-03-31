# Marketing Incrementality & Customer Behavior Intelligence Platform

An end-to-end applied machine learning and causal inference project built on the Olist Brazilian E-Commerce + Marketing Funnel dataset to analyze customer behavior across digital and marketing channels, measure campaign effectiveness, and predict conversion outcomes.

This project was designed to mirror the responsibilities of a production-facing Marketing Science / Applied ML role: translating business questions into statistical frameworks, building scalable analytical pipelines, deploying predictive models, and connecting insights to customer acquisition, engagement, conversion, retention, and marketing efficiency.

---

## Business Problem

Marketing teams often struggle to answer questions such as:

- Which customer behaviors are most predictive of conversion?
- Which channels are associated with stronger acquisition and retention outcomes?
- Are campaigns driving **incremental lift**, or are they just correlated with high-intent users?
- How can we identify likely converters earlier and allocate spend more efficiently?

This project addresses those questions by combining:
- customer journey analytics
- predictive modeling
- experiment evaluation
- causal inference
- API-based model serving

---

## Project Objectives

This project was built to demonstrate the ability to:

- Develop statistical and machine learning models to analyze customer behavior
- Investigate drivers of acquisition, engagement, conversion, and retention
- Design and evaluate marketing experiments / A/B tests
- Apply causal inference methods to estimate incremental campaign impact
- Build reproducible analytical pipelines using Python and SQL
- Deploy production-style prediction services
- Translate technical findings into business recommendations

---

## Dataset

### Primary Sources
- **Olist Brazilian E-Commerce Dataset**
- **Olist Marketing Funnel Dataset**

### Core Data Used
- customers
- orders
- order items
- order payments
- reviews
- products
- marketing qualified leads
- closed deals

### Enrichment Layer
To simulate a more realistic marketing analytics environment, the project generates additional analytical tables such as:

- `marketing_touchpoints`
- `campaigns`
- `customer_sessions`
- `experiment_assignments`
- `attribution_summary`
- `customer_feature_mart`

These tables enable channel-level analysis, experiment simulation, and predictive feature generation.

---

## End-to-End Workflow

### 1) Data Engineering & Feature Pipeline
Built a scalable feature generation workflow using Python + SQL-style transformations to create customer-level training data from transactional and marketing records.

Examples of engineered features:
- order frequency
- total spend
- average order value
- review behavior
- touchpoint volume
- email / paid click counts
- recency of marketing exposure
- first-touch / last-touch channel features

### 2) Predictive Modeling
Trained predictive models for:

- **Conversion Propensity**
  - probability a customer converts based on transactional and marketing behavior

- **Retention / Repeat Purchase**
  - likelihood of repeat engagement / retained purchasing behavior

Models were trained using:
- scikit-learn
- XGBoost / LightGBM compatible pipeline design
- structured feature preprocessing
- reproducible training workflow

### 3) Experimentation & A/B Test Analysis
Created a treatment/control evaluation framework to measure simulated campaign lift and compare:

- conversion rate
- average order value
- revenue per exposed customer

### 4) Causal Inference
Implemented an inverse propensity weighting (IPTW) workflow to estimate **incremental campaign impact** and distinguish:

- observational correlation
- causal treatment effect

### 5) Production Deployment
Deployed a production-style prediction API using:

- **FastAPI**
- **Docker**
- serialized trained models
- structured request/response schema

This simulates how a marketing or growth team could score customers in real time for downstream targeting or prioritization.

---

## Tech Stack

### Languages / Analytics
- Python
- SQL

### ML / Statistics
- pandas
- NumPy
- scikit-learn
- XGBoost / LightGBM-ready architecture
- causal inference workflow (IPTW)

### Deployment / Engineering
- FastAPI
- Docker
- joblib
- pytest
- modular project structure
- version-controlled workflow

### Cloud / Production Alignment
Project architecture was designed to align with:
- Amazon Redshift-style analytical workflows
- AWS SageMaker deployment patterns
- production ML / analytics engineering workflows

---

## Project Structure

```bash
marketing-intelligence-platform/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── sql/
├── src/
│   └── marketing_intelligence/
├── tests/
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

---

## Key Results

### Customer Behavior & Funnel Analytics
- Built a customer-level analytical feature mart spanning **100k+ orders**, **90k+ customers**, and multiple synthetic marketing funnel touchpoints
- Generated reusable customer features for acquisition, conversion, and retention modeling
- Identified measurable variation in conversion behavior across recency, order frequency, spend, and marketing exposure patterns

### Predictive Modeling
- Developed an end-to-end **conversion propensity scoring pipeline** with deployable inference workflow
- Produced customer-level conversion probabilities via API for downstream targeting and campaign prioritization
- Built retention-oriented scoring logic to support repeat purchase / customer lifecycle analysis

### Experimentation & A/B Test Results
Evaluated a simulated treatment/control campaign experiment across **99,441 customers**:

| Variant   | Customers | Conversions | Conversion Rate | Avg Revenue |
|-----------|-----------|-------------|-----------------|-------------|
| Control   | 49,860    | 3,473       | **6.97%**       | **160.02**  |
| Treatment | 49,581    | 3,557       | **7.17%**       | **160.95**  |

#### Experiment Summary
- **Absolute conversion lift:** **+0.21 percentage points**
- **Relative conversion lift:** **~3.0%**
- **z-statistic:** **-1.28**
- **p-value:** **0.199**

#### Interpretation
The treatment group showed a **directionally positive improvement** in conversion rate compared with control, but the effect was **not statistically significant** at the 95% confidence level. This indicates the observed lift could plausibly be explained by random variation rather than a robust campaign effect.

### Causal Inference Results (IPTW)
To distinguish observational lift from true incremental effect, I applied **Inverse Probability of Treatment Weighting (IPTW)**:

| Metric | Value |
|--------|-------|
| Naive control conversion rate | **6.97%** |
| Naive treatment conversion rate | **7.17%** |
| Naive observed lift | **+0.21 pp** |
| IPTW-adjusted control rate | **7.03%** |
| IPTW-adjusted treatment rate | **7.10%** |
| IPTW-adjusted ATE | **+0.08 pp** |

#### Causal Interpretation
After adjusting for confounding and treatment selection bias, the estimated treatment effect dropped from **+0.21 percentage points** to **+0.08 percentage points**.

This showed that the **naive observed lift overstated incremental campaign impact by ~2.7x**, demonstrating why causal adjustment is important in marketing measurement and budget allocation decisions.

### Production Readiness
- Packaged the workflow into a reproducible ML project with:
  - modular training pipeline
  - model serialization
  - API deployment
  - testable project structure
- Exposed model inference through FastAPI endpoints for **real-time scoring use cases**

---

## Example API Output

Example inference response:

```json
{
  "conversion_probability": 0.000014252817891247105,
  "prediction": 0
}
```

### Interpretation
This indicates the model estimates a **very low probability of conversion** for the provided customer profile. This is expected in imbalanced marketing datasets where conversion events are relatively rare.

In practice, these probabilities are useful for:
- customer prioritization
- audience segmentation
- treatment eligibility rules
- campaign targeting workflows

---

## Business Value

This project demonstrates how ML and statistical analysis can support decisions such as:

- which customers to prioritize for re-engagement
- which segments are likely to convert
- which campaigns may be incrementally effective
- where marketing spend may be inefficient
- how analytical outputs can connect to revenue and cost optimization

---

## What This Project Demonstrates

This project was intentionally designed to reflect real-world responsibilities in Marketing Science / Applied ML roles, including:

- customer behavior analysis
- statistical modeling
- experiment design and measurement
- causal inference
- production-style model deployment
- scalable analytical pipeline design
- translating technical findings into business outcomes

---

## Future Improvements

Potential production upgrades include:

- probability calibration
- uplift modeling
- better class imbalance handling
- Redshift-native SQL orchestration
- SageMaker endpoint deployment
- MLflow experiment tracking
- model monitoring and drift detection
- budget allocation optimization

---

## How to Run

```bash
pip install -r requirements.txt
```

### Validate raw files
```bash
$env:PYTHONPATH="src"
python -m marketing_intelligence.cli validate --data-dir data/raw
```

### Build features
```bash
$env:PYTHONPATH="src"
python -m marketing_intelligence.cli build-features --project-root .
```

### Train models
```bash
$env:PYTHONPATH="src"
python -m marketing_intelligence.cli train --project-root .
```

### Run experiment / causal analysis
```bash
$env:PYTHONPATH="src"
python -m marketing_intelligence.cli analyze --project-root .
```

### Start API
```bash
$env:PYTHONPATH="src"
uvicorn marketing_intelligence.api:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

---


