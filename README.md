# MRTB2173-51-CaseStudy

## Telecommunication Customer Churn Risk Prediction

This repository contains an assessed Agile Data Science case study for
MRTB2173. The project uses the public Telco Customer Churn dataset to
construct and iteratively refine a customer-retention decision-support
MVP.

## Problem

Telecommunications customers may discontinue their subscriptions
without being identified early. The project proposes a churn-risk
application that uses historical customer, contract, service, and
billing information to help an authorized retention team prioritize
review.

## Dataset

- Dataset: Telco Customer Churn
- Source: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Expected file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Raw data is preserved in `data/raw/`.
- Cleaned and engineered data are stored separately.

## Primary Users

- Customer Retention Manager
- Customer Service Supervisor
- Business Manager
- Data Science Team
- IT and DataOps Team

## User Stories

1. As a Customer Retention Manager, I want a ranked list of customers
   by churn risk so that I can prioritize retention activities.
2. As a Customer Service Supervisor, I want to view the main factors
   associated with each high-risk customer so that agents can select a
   suitable intervention.
3. As a Business Manager, I want a summary of churn patterns by
   customer segment so that I can allocate retention resources more
   effectively.

## Agile Sprint Development

### Sprint 1 - Data Foundation

Deliverables:

- cleaned CSV and Excel dataset
- data dictionary
- before-and-after data-quality report
- churn summaries
- exploratory charts
- findings summary

### Sprint 2 - Prediction MVP

Deliverables:

- engineered features
- class-weighted logistic-regression baseline
- initial probability and risk table
- separate high-risk customer CSV

### Sprint 3 - Evaluation and Refinement

Deliverables:

- Logistic Regression and Random Forest comparison
- five-fold cross-validation metrics
- precision, recall, F1, ROC-AUC, and confusion matrices
- threshold analysis
- model-importance evidence
- refined risk table
- preserved Sprint 2 comparison

### Sprint 4 - Deployment Planning

Deliverables:

- internal Streamlit prototype
- saved scikit-learn pipeline
- automated pytest tests
- GitHub Actions CI workflow
- deployment architecture
- monitoring plan
- deployment and governance documentation

## Project Structure

```text
MRTB2173-51-CaseStudy/
├── .github/workflows/
│   └── ci_cd_pipeline.yml
├── charts/
│   ├── sprint1/
│   ├── sprint2/
│   ├── sprint3/
│   └── sprint4/
├── dashboard/
│   └── app.py
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   ├── agile_case_study_plan.xlsx
│   ├── deployment_proposal.md
│   └── slide_evidence_checklist.md
├── models/
│   ├── churn_risk_pipeline.joblib
│   └── model_metadata.json
├── outputs/
│   ├── sprint1/
│   ├── sprint2/
│   ├── sprint3/
│   └── sprint4/
├── src/
│   └── pipeline_utils.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Run the Automated Tests

```bash
pytest tests/ -v
```

## Run the Streamlit Prototype

```bash
streamlit run dashboard/app.py
```

## Streamlit Cloud Settings

- Repository: `laizizheng-glitch/MRTB2173-51-CaseStudy`
- Branch: `main`
- Main file path: `dashboard/app.py`

The coursework requires deployment thinking rather than a completed
organizational deployment. Streamlit Cloud can be used only as an
optional classroom demonstration. Real customer information should not
be uploaded to a public application.

## Model and Interpretation Notes

- Gender is retained in the cleaned dataset but excluded from model
  training by default.
- Risk probabilities support prioritization and do not guarantee
  customer behaviour.
- The observed risk-factor text is rule based and is not a causal
  explanation.
- Model evaluation is based on the held-out test set and five-fold
  cross-validation.
- Business benefit would require later comparison with actual
  retention outcomes.

## Feedback Requirement

The workbook `docs/agile_case_study_plan.xlsx` contains an empty
feedback log. Replace the placeholders with genuine peer, lecturer, or
stakeholder review evidence. Do not present example comments as actual
feedback.

## Agile and DataOps Alignment

- Short sprints deliver incremental value.
- The backlog keeps technical tasks tied to user needs.
- Sprint reviews collect feedback on reviewable outputs.
- Retrospectives identify process improvements.
- Reusable feature logic reduces transformation inconsistency.
- Git provides version control.
- pytest and GitHub Actions provide rapid automated feedback.
- The monitoring plan extends the project beyond initial model
  training.

## Deployment Status

Conceptual proof of concept only. No production organizational system
has been deployed.
