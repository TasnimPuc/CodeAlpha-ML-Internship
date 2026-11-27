# CodeAlpha – Machine Learning Internship

![repo health](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/TasnimPuc/bd-rooftop-solar-calculator/main/.github/health/badge.json)

Four machine-learning projects built during the CodeAlpha ML internship: classical ML on tabular data, audio deep learning, and computer vision. Every project is a Kaggle notebook with all outputs saved, plus a README explaining the approach and results.

| # | Project | What it does | Techniques | Key result |
|---|---|---|---|---|
| 1 | [Credit Scoring Model](01_CreditScoringModel/) | Predicts whether a credit-card client will default next month | Feature engineering, Logistic Regression, Decision Tree, tuned Random Forest | ROC-AUC **0.780** · F1 0.545 (30k clients) |
| 2 | [Emotion Recognition from Speech](02_EmotionRecognitionFromSpeech/) | Recognises 8 emotions from voice recordings | MFCC features, audio augmentation, CNN + BiLSTM | **85.0 %** test accuracy (RAVDESS + TESS) |
| 3 | [Handwritten Character Recognition](03_HandwrittenCharacterRecognition/) | Reads handwritten characters and full words | CNN, CRNN (CNN + BiLSTM + CTC) | **89.9 %** on 47-class EMNIST · 78.4 % word accuracy |
| 4 | [Disease Prediction](04_DiseasePrediction/) | Predicts heart disease, diabetes and breast cancer from patient data | SVM, Logistic Regression, Random Forest, XGBoost, GridSearchCV | Breast cancer **98.2 %** · Heart **84.8 %** (recall 0.92) |

## Repository structure
```
├── 01_CreditScoringModel/
├── 02_EmotionRecognitionFromSpeech/
├── 03_HandwrittenCharacterRecognition/
└── 04_DiseasePrediction/
    ├── CodeAlpha_<Project>.ipynb   # notebook with outputs (run on Kaggle)
    ├── README.md                   # problem, dataset, method, results
    ├── requirements.txt
    └── results/                    # plots and metrics
```

## Run any project
1. Kaggle → **New Notebook** → **File → Import Notebook** → upload the `.ipynb`.
2. **+ Add Input** → add the dataset listed in that project's README.
3. For projects 2 and 3: **Settings → Accelerator → GPU**.
4. **Run All**.

## Tech stack
Python · pandas · NumPy · scikit-learn · XGBoost · TensorFlow/Keras · librosa · matplotlib · Kaggle
