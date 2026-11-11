# Health-Project 3: Breast Cancer Detection

## 1. Problem Statement
Diagnostic labs require an automated machine learning solution to cross-check pathologist readings of Fine-Needle Aspirate (FNA) biopsy measurements by flagging likely-malignant tumours for priority review.

## 2. ML Formulation
* **Type:** Binary Classification
* **Target Variable:** diagnosis (0 = Benign, 1 = Malignant)
* **Features:** 30 numeric measurements computed from digitized images of FNA biopsy samples.

## 3. Success Metrics & KPIs
* **Primary Metric:** High Recall on the Malignant class (1) to minimize False Negatives.
* **Secondary Metrics:** Precision, F1-Score, ROC-AUC.
