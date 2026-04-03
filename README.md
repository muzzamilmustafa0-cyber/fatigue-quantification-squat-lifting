# EMG Fatigue Quantification — ML Pipeline

A machine learning and deep learning pipeline for EMG-based fatigue classification and Borg RPE regression during squat lifting tasks.

---

## Repository Structure

```
.
├── analysis/
│   ├── run_all.py               # Single entry-point: runs the full pipeline
│   ├── data_loader.py           # Parse raw EMG files + label spreadsheet
│   ├── signal_processing.py     # Butterworth bandpass filter, rectification, linear envelope
│   ├── feature_extraction.py    # Time/frequency domain features (RMS, MPF, ZCR, AR, etc.)
│   ├── dataset_builder.py       # Align features with RPE labels → ML-ready CSV
│   ├── traditional_ml.py        # LR, SVM, DT, RF, XGBoost, LightGBM (LOSO + 10-fold CV)
│   ├── deep_learning.py         # MLP, 1D-CNN, BiLSTM (TensorFlow)
│   ├── results_tables.py        # Generate CSV summary tables
│   ├── visualizations.py        # Publication-quality figures (300 DPI)
│   └── requirements.txt         # Python dependencies
├── README.md
└── .gitignore
```

---

## Requirements

- Python 3.11+

Install dependencies:
```bash
pip install -r analysis/requirements.txt
```

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.24 | Numerical arrays |
| pandas | ≥2.0 | Data handling |
| scikit-learn | ≥1.3 | Traditional ML models |
| xgboost | ≥1.7 | XGBoost classifier/regressor |
| lightgbm | ≥4.0 | LightGBM classifier/regressor |
| tensorflow | 2.21.0 | MLP, 1D-CNN, BiLSTM |
| shap | ≥0.43 | SHAP feature importance |
| matplotlib | ≥3.7 | Figures |
| seaborn | ≥0.13 | Heatmaps |
| scipy | ≥1.11 | Signal processing, statistical tests |
| openpyxl | ≥3.1 | Excel file reading |

---

## Data Requirements (Not Included)

To run the pipeline you need to provide (not included in this repository):

1. **`Data/`** — Per-subject Excel files with raw MegaWin ASCII EMG time-series (8 channels, 1020 Hz).
2. **`FINAL DATA ANALYSIS FORM.xlsx`** — Workbook with per-observation Borg RPE, heart rate, and biomechanical features.

Place these in the project root directory (same level as `analysis/`).

---

## Running the Pipeline

```bash
cd analysis
python run_all.py
```

Steps executed:
1. **Dataset Building** — Load raw EMG, extract features, build classification and DL datasets
2. **Traditional ML** — Train and evaluate all models under LOSO and 10-fold CV
3. **Deep Learning** — Train MLP, 1D-CNN, BiLSTM
4. **Figures** — Generate all figures to `analysis/figures/`

Results are saved to `analysis/results/` (CSV tables) and `analysis/figures/` (PNG/PDF figures).

---

## Reproducibility

All random seeds are fixed (`seed=42`):
- Train/test splits and ensemble models (`random_state=42`)
- TensorFlow (`tf.random.set_seed(42)`)
- No oversampling (e.g. SMOTE) is applied in the reported analyses

---

## License

The code will be released under the **MIT License** upon publication.
