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

---

## Random Forest Model Results

After establishing the logistic regression baselines, two Random Forest models were trained and evaluated:

1. Standard Random Forest
2. Balanced Random Forest with `class_weight="balanced"`

Random Forest was added because machine failure prediction may depend on nonlinear relationships and interactions among operating variables such as torque, rotational speed, tool wear, power demand, and temperature behavior. Unlike logistic regression, Random Forest can capture these nonlinear patterns more effectively.

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9695 | 0.6522 | 0.2206 | 0.3297 | 0.9263 |
| Balanced Logistic Regression | 0.8555 | 0.1760 | 0.8824 | 0.2934 | 0.9321 |
| Random Forest | 0.9905 | 0.9455 | 0.7647 | 0.8455 | 0.9659 |
| Balanced Random Forest | 0.9880 | 0.9400 | 0.6912 | 0.7966 | 0.9718 |

The results show that Random Forest significantly outperformed the logistic regression baselines. The standard Random Forest model achieved the strongest overall balance between precision, recall, and F1-score.

The standard logistic regression model was too conservative and missed many actual failures. The balanced logistic regression model improved recall, but its low precision showed that it produced many false alarms. In contrast, the Random Forest model achieved both high precision and strong recall, making it more practical for predictive maintenance.

Although the balanced Random Forest model achieved the highest ROC-AUC score, the standard Random Forest model had the best F1-score and a better practical balance between detecting real failures and avoiding unnecessary alerts.

---

## Current Best Model

Based on the model comparison results, the standard Random Forest model is currently selected as the leading model for PredictWise.

The model achieved:

- Accuracy: 0.9905
- Precision: 0.9455
- Recall: 0.7647
- F1-score: 0.8455
- ROC-AUC: 0.9659

This model provides the best balance across the most important evaluation metrics. For predictive maintenance, this balance is important because the system needs to detect actual failures while avoiding excessive false alarms.

The strong performance of Random Forest suggests that machine failure prediction in this dataset depends on nonlinear relationships and feature interactions. This supports the use of tree-based models for industrial failure prediction tasks.

---

## Threshold Tuning

Threshold tuning was used to evaluate how different classification cutoffs affect the Random Forest model's precision, recall, and F1-score.

By default, classification models usually use a threshold of 0.5. This means that if the predicted failure probability is greater than or equal to 0.5, the model predicts machine failure. However, in predictive maintenance, the default threshold may not always be the most practical choice because machine failures are rare but costly.

The threshold tuning results showed a clear trade-off:

- Lower thresholds increase recall and detect more actual failures, but they also create more false alarms.
- Higher thresholds increase precision and reduce false alarms, but they may miss more actual failures.

The best F1-score occurred around a threshold of 0.4, suggesting that this threshold provides the strongest balance between failure detection and false alarm control.

For PredictWise, a threshold near 0.4 is currently the strongest candidate because it balances the cost of missed failures against the cost of unnecessary maintenance alerts.

---

## Precision-Recall Analysis

Because the dataset is highly imbalanced, precision-recall analysis is more informative than accuracy alone.

The Random Forest model achieved an Average Precision score of 0.881, indicating strong performance on the minority failure class across different classification thresholds.

The precision-recall curve shows that the model can maintain relatively strong precision while achieving useful recall. This is important for predictive maintenance because the model must identify rare failure events without overwhelming maintenance teams with excessive false alarms.

This analysis reframes PredictWise as a decision-support system rather than a fixed classifier. In a real manufacturing environment, the classification threshold could be adjusted depending on maintenance capacity, downtime cost, and risk tolerance.

---

## Model Interpretation

Random Forest feature importance was used to understand which variables contributed most to machine failure prediction.

The most important feature was `Power proxy [W]`, followed by `Torque [Nm]`, `Rotational speed [rpm]`, `Temperature difference [K]`, and `Tool wear [min]`.

| Feature | Importance |
|---|---:|
| Power proxy [W] | 0.2500 |
| Torque [Nm] | 0.1796 |
| Rotational speed [rpm] | 0.1775 |
| Temperature difference [K] | 0.1405 |
| Tool wear [min] | 0.1171 |
| Air temperature [K] | 0.0528 |
| Process temperature [K] | 0.0460 |
| Tool wear level_High | 0.0134 |
| Type_L | 0.0121 |
| Type_M | 0.0085 |
| Tool wear level_Medium | 0.0027 |

The high importance of `Power proxy [W]` suggests that the interaction between torque and rotational speed is especially useful for predicting machine failure. This supports the engineering assumption that failure risk is strongly related to mechanical power demand and operating intensity.

`Torque [Nm]` and `Rotational speed [rpm]` were also among the most important variables, showing that mechanical load and operating speed are major contributors to failure prediction.

`Temperature difference [K]` ranked higher than the original raw temperature variables, suggesting that engineered features can capture machine condition more effectively than raw sensor values alone.

`Tool wear [min]` also contributed meaningfully to the model, which aligns with the maintenance expectation that accumulated tool usage increases failure risk.

Product type and categorical tool wear level variables had relatively low importance compared with continuous operating variables. This suggests that machine failure prediction in this dataset is driven more by operating conditions than by product category.

Overall, the feature importance analysis supports the engineering interpretation that machine failure risk is influenced by mechanical stress, power demand, thermal load, and accumulated wear.

---

## Updated Project Status

- [x] Project topic selected
- [x] Dataset downloaded and loaded
- [x] Basic data inspection completed
- [x] Missing value check completed
- [x] Initial EDA visualizations completed
- [x] Failure rate analysis by product type
- [x] Feature engineering
- [x] Modeling dataset preparation
- [x] Baseline logistic regression model
- [x] Balanced logistic regression model
- [x] Random Forest model
- [x] Balanced Random Forest model
- [x] Model comparison table
- [x] Threshold tuning
- [x] Precision-recall analysis
- [x] Feature importance analysis
- [x] Model interpretation
- [x] Save final visualizations to assets folder
- [x] Build Streamlit prediction app
- [x] Final README and project documentation

---

## Updated Next Steps

The next stage of the project will focus on turning the model into a simple interactive decision-support tool.

Planned next steps:

- Save key visualizations to the `assets/` folder
- Build a Streamlit app for PredictWise
- Allow users to input machine operating conditions
- Output predicted failure probability
- Display risk level based on the selected threshold
- Provide simple maintenance recommendations
- Save the trained Random Forest model for app deployment
- Finalize project documentation and screenshots
