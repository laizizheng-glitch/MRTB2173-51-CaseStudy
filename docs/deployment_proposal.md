# Sprint 4 Deployment Proposal

## Proposed Application

An internal Streamlit web application would display customer churn
probabilities, Low/Medium/High risk categories, observed risk factors,
segment summaries, filters, and a downloadable prioritized list.

## Intended Users

- Customer Retention Manager
- Customer Service Supervisor
- Approved Business Managers
- Data Science and DataOps Team

## Proposed Weekly Workflow

1. An authorized customer CSV is placed in a controlled location.
2. Automated validation checks expected columns, duplicates, and
   missing values.
3. The shared cleaning and feature-engineering functions transform the
   data.
4. The saved `Random Forest` pipeline generates probabilities.
5. Probabilities are converted into configurable Low, Medium, and High
   risk categories.
6. The internal Streamlit application displays masked customer
   references.
7. Authorized users export a filtered list for operational review.
8. Monitoring results and user feedback create new backlog items.

## Possible Platform

- Interface: Streamlit
- Data processing: Python and pandas
- Model: scikit-learn
- Model storage: joblib
- Version control: GitHub
- Automated tests: pytest
- Continuous Integration: GitHub Actions
- Hosting: private organizational server or private cloud environment
- Refresh approach: weekly batch scoring

## Security and Governance

- Restrict access to approved users.
- Use masked customer identifiers in the user interface.
- Do not publish real customer data on a public demo site.
- Record data, code, model, and threshold versions.
- Maintain an audit trail for scoring outputs.
- Assign ownership for data quality, model review, and incident
  response.
- Review whether sensitive or protected attributes should be excluded.

## Monitoring

Refer to:

`outputs/sprint4/sprint4_monitoring_plan.csv`

The proposed monitoring covers data quality, duplicate IDs, prediction
distribution, model metrics, job success, and business use.

## Deployment Status

This repository packages a runnable prototype and deployment proposal.
It does not claim that the model has been deployed into a live
telecommunications organization.
