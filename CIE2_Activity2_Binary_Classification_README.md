# CIE 2 — Activity 2: Binary Classification Using Logistic Regression

**Course:** Regression and Classification Algorithms (T.Y. B.Tech AI & DS, Semester V, AY 2026–27)  
**Institution:** Zeal College of Engineering and Research (ZCOER), Pune  
**Submission GitHub Reference:** [https://github.com/complexiota](https://github.com/complexiota)

---

## 1. Project Overview
This repository presents an end-to-end binary classification workflow built from the perspective of a clinical data scientist supporting breast cancer diagnostic screening. Using **Logistic Regression**, the pipeline predicts whether a fine-needle aspirate (FNA) breast mass biopsy is **Malignant (`1`, Positive Class)** or **Benign (`0`, Negative Class)**, analyzes probability calibration across decision thresholds, and recommends an operationally safe screening threshold.

## 2. Dataset & Source Citation
* **Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset (WDBC)
* **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (included locally as `breast_cancer_wisconsin.csv` for offline reproducibility)
* **Dimensions:** `569` patient records × `31` columns (`30` continuous morphological features + `1` binary target `diagnosis`)
* **Class Distribution:** `357` Benign (`62.74%`) and `212` Malignant (`37.26%`) — `0` missing values, `0` duplicate records.

## 3. Repository Structure
```text
├── CIE2_Activity2_Binary_Classification.ipynb   # Executed Jupyter Notebook with full analysis, plots & decision log
├── breast_cancer_wisconsin.csv                  # Clean dataset CSV (0 = Benign, 1 = Malignant)
├── CIE2_Activity2_Report.docx                   # Formatted submission report with embedded visuals & metric tables
├── requirements.txt                             # Python package dependencies
├── figures/                                     # High-resolution exported plots
│   ├── 01_class_distribution.png
│   ├── 02_sigmoid_curve.png
│   ├── 03_confusion_matrix_and_roc.png
│   └── 04_decision_behaviour.png
└── README.md                                    # Project documentation and key results
```

## 4. Key Results & Threshold Analysis (Unseen Test Set, n = 114)

| Metric | Formula | Default Threshold (`0.50`) | Recommended Threshold (`0.30`) | Clinical Interpretation |
| :--- | :--- | :---: | :---: | :--- |
| **Accuracy** | `(TP+TN)/(TP+TN+FP+FN)` | `0.9649` (96.49%) | **`0.9825` (98.25%)** | Beats the 62.74% majority-class baseline by +35.51%. |
| **Precision** | `TP/(TP+FP)` | `0.9750` (97.50%) | **`0.9762` (97.62%)** | 41 of 42 patients flagged for biopsy are truly malignant. |
| **Recall (Sensitivity)** | `TP/(TP+FN)` | `0.9286` (92.86%) | **`0.9762` (97.62%)** | Detects 41 of 42 malignant cases (only 1 missed case). |
| **Specificity** | `TN/(TN+FP)` | `0.9861` (98.61%) | **`0.9861` (98.61%)** | Correctly clears 71 of 72 benign patients. |
| **F1-Score** | `2*(Prec*Rec)/(Prec+Rec)` | `0.9512` (95.12%) | **`0.9762` (97.62%)** | Optimal balance between sensitivity and precision. |
| **ROC-AUC** | Area under ROC curve | **`0.9960`** | **`0.9960`** | Near-perfect class ranking ability across all thresholds. |

### Threshold Sensitivity Summary
* **Threshold `0.30` (Recommended):** `42` predicted positive (`TP=41, FP=1, FN=1, TN=71`) — cuts missed cancers (`FN`) by 67% compared to `0.50` without adding any false alarms.
* **Threshold `0.50` (Default):** `40` predicted positive (`TP=39, FP=1, FN=3, TN=71`).
* **Threshold `0.70` (Strict):** `38` predicted positive (`TP=38, FP=0, FN=4, TN=72`) — achieves 100% precision but misses 4 malignant tumors, making it clinically unsafe for primary screening.

## 5. How to Run Locally
```bash
git clone https://github.com/<your-username>/CIE2-Activity2-Binary-Classification.git
cd CIE2-Activity2-Binary-Classification
pip install -r requirements.txt
jupyter notebook CIE2_Activity2_Binary_Classification.ipynb
```

## 6. Ethical Note & Limitations
Model outputs are probability-based risk estimates rather than deterministic medical diagnoses. This classifier should be used exclusively as a clinical decision-support triage tool under physician supervision.
