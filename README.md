# PredictWise: Predictive Maintenance System

## Project Overview

PredictWise is a machine learning project designed to predict industrial machine failures using manufacturing sensor data. In modern manufacturing systems, unexpected equipment failures can cause production downtime, quality issues, maintenance cost increases, and delivery delays. Traditional maintenance strategies often rely on either reactive maintenance, where machines are repaired after failure, or scheduled maintenance, where machines are serviced at fixed intervals regardless of their actual condition.

This project explores a data-driven predictive maintenance approach. By analyzing machine operating conditions such as temperature, rotational speed, torque, and tool wear, the goal is to identify failure patterns and build a model that can estimate whether a machine is likely to fail.

PredictWise connects mechanical engineering, manufacturing systems, and machine learning. It is part of a broader smart manufacturing portfolio focused on using data and AI to improve factory reliability, productivity, and decision-making.

---

## Engineering Problem

Unplanned machine failure is a major challenge in manufacturing environments. A single machine failure can interrupt production flow, reduce overall equipment effectiveness, increase scrap rate, and create additional maintenance costs.

The central engineering question of this project is:

> Can machine operating data be used to predict equipment failure before it happens?

Instead of only analyzing failures after they occur, PredictWise aims to move toward predictive decision-making. The system is designed to help maintenance engineers, reliability engineers, and production managers identify high-risk operating conditions and take preventive action earlier.

---

## Objectives

The main objectives of this project are:

- Explore and understand industrial machine sensor data
- Analyze the relationship between operating conditions and machine failure
- Identify important failure-related features such as tool wear, torque, rotational speed, and temperature behavior
- Engineer new features that better represent machine operating conditions
- Build baseline machine learning models for binary failure prediction
- Evaluate model performance using appropriate classification metrics
- Develop interpretable insights that can support maintenance decisions

---

## Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset**, a synthetic industrial dataset designed for predictive maintenance analysis.

The dataset contains **10,000 production records** and includes product information, machine operating variables, failure labels, and specific failure type indicators.

### Key Variables

| Variable | Description |
|---|---|
| `UDI` | Unique record ID |
| `Product ID` | Product identifier |
| `Type` | Product quality type: L, M, or H |
| `Air temperature [K]` | Ambient air temperature in Kelvin |
| `Process temperature [K]` | Process temperature in Kelvin |
| `Rotational speed [rpm]` | Machine rotational speed |
| `Torque [Nm]` | Machine torque/load |
| `Tool wear [min]` | Accumulated tool wear time |
| `Machine failure` | Binary target variable: 0 = no failure, 1 = failure |
| `TWF` | Tool Wear Failure |
| `HDF` | Heat Dissipation Failure |
| `PWF` | Power Failure |
| `OSF` | Overstrain Failure |
| `RNF` | Random Failure |

For the first stage of this project, `Machine failure` is used as the primary prediction target. The specific failure type indicators will be used for exploratory analysis and may be extended into a multi-class or multi-label failure classification task in future work.

---

## Planned Workflow


Industrial Sensor Data
        ↓
Data Cleaning and Inspection
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Machine Learning Model Development
        ↓
Failure Prediction
        ↓
Maintenance Recommendation


## Baseline Model Results

Two logistic regression baseline models were developed for binary machine failure prediction:

1. Standard Logistic Regression
2. Balanced Logistic Regression with `class_weight="balanced"`

Because the dataset is highly imbalanced, with only about 3.39% failure cases, model performance was evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix analysis.

### Standard Logistic Regression

The standard logistic regression model achieved high overall accuracy, but its failure-class recall was low. This means the model was conservative when predicting failures. It produced fewer false alarms, but it missed many actual machine failure cases.

### Balanced Logistic Regression

The balanced logistic regression model significantly improved failure-class recall by assigning more weight to the minority failure class during training. This allowed the model to detect most actual failure cases. However, this improvement came with lower precision, meaning the model also produced more false alarms.

### Key Insight

The baseline models show a clear trade-off between precision and recall. In predictive maintenance, this trade-off is important because missing a real machine failure can lead to unplanned downtime, while too many false alarms can increase unnecessary maintenance work.

At this stage, the goal is not to produce a perfect model, but to establish a baseline and understand the modeling challenge. The next stage will compare logistic regression with tree-based models such as Random Forest, which may better capture nonlinear interactions among operating variables.

### Next Steps

- Train and evaluate Random Forest models
- Compare standard and class-weighted approaches
- Create a model comparison table
- Analyze feature importance
- Explore threshold tuning to control the precision-recall trade-off
- Add precision-recall curve analysis
