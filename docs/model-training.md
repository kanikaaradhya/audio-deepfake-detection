# Model Training — Audio DeepFake Detection

This document covers dataset preparation, model selection, ensemble strategy, and evaluation results.

---

## Dataset: Fake-or-Real (for-2sec)

The [FoR dataset](https://bil.eecs.yorku.ca/datasets/) by Reimao & Tzerpos (York University) is the standard benchmark for synthetic speech detection. We used the **for-2sec** variant:

| Property | Value |
|----------|-------|
| Real clips | 6,978 |
| Fake clips | 6,978 |
| Total | 13,956 |
| Duration per clip | 2 seconds (fixed) |
| Sample rate | 16 kHz (resampled to 22,050 Hz during extraction) |
| Fake sources | DeepVoice 3, Google Cloud TTS, WaveNet, Amazon Polly, Microsoft Azure TTS, Baidu Cloud TTS |
| Real sources | CMU Arctic, LJSpeech, VoxForge, original recordings |

**Why for-2sec:** the original FoR dataset has variable-length clips (real speech averages 5 seconds; synthetic averages 2.3 seconds). A model trained on variable-length audio could learn to classify based on duration alone — a shortcut that wouldn't generalise to real-world use. The for-2sec variant truncates everything to exactly 2 seconds, forcing the model to learn acoustic differences rather than length differences.

**Train/test split:** 80/20 stratified split, preserving the 50/50 real/fake balance in both sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Train: 11,164 samples | Test: 2,792 samples
```

## Model Selection

### Why Gradient-Boosted Trees?

For tabular feature vectors (121 dimensions, all numeric), gradient-boosted trees consistently outperform other approaches:
- **vs. Random Forest:** boosting sequentially corrects errors; bagging only reduces variance. On structured features, boosting wins.
- **vs. SVM:** SVMs require careful kernel and regularisation tuning. XGBoost/LightGBM handle feature interactions natively and scale better.
- **vs. Deep learning:** a neural network on a 121-dim input with 11,164 training samples would overfit without heavy regularisation. GBTs handle this scale naturally.

### XGBoost Configuration

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric='logloss',
    random_state=42
)
```

- **200 trees, depth 6:** enough capacity for 121 features without overfitting on ~11K samples
- **learning_rate=0.1:** standard — aggressive enough to converge in 200 rounds, conservative enough to avoid overshooting

### LightGBM Configuration

```python
from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=-1
)
```

Same hyperparameters for fair comparison. The key difference is internal: LightGBM uses **leaf-wise** tree growth (grows the leaf with the highest loss reduction), while XGBoost uses **level-wise** (grows all leaves at the same depth). This means they learn slightly different decision boundaries even with identical configs.

## Ensemble Strategy

```python
from sklearn.ensemble import VotingClassifier

ensemble = VotingClassifier(
    estimators=[('xgb', xgb), ('lgbm', lgbm)],
    voting='soft'
)
```

**Soft voting** averages the predicted probabilities from both models, then picks the class with the higher average probability. This is more nuanced than hard voting (which just counts votes), because it accounts for how *confident* each model is.

Example: if XGBoost says 90% fake and LightGBM says 60% fake, soft voting averages to 75% fake — which is more informative than "2 votes for fake."

## Results

```
========================================
Accuracy:  0.9982
F1 Score:  0.9982
========================================

              precision    recall  f1-score   support

        Real       1.00      1.00      1.00      1396
        Fake       1.00      1.00      1.00      1396

    accuracy                           1.00      2792
   macro avg       1.00      1.00      1.00      2792
weighted avg       1.00      1.00      1.00      2792
```

**99.82% accuracy** on 2,792 held-out test samples — 5 misclassifications out of 2,792.

The rounded precision and recall both show 1.00 because the actual values are ≥0.995 (fewer than 3 errors per class out of 1,396).

## Comparison with Literature

| Paper / Method | Dataset | Accuracy |
|---------------|---------|----------|
| Iqbal et al. — SVM | FoR-2sec | 93.0% |
| Iqbal et al. — XGBoost | FoR-norm | 97.0% |
| **This project — XGB + LGBM ensemble** | **FoR-2sec** | **99.82%** |

The ensemble approach with richer feature engineering (121 dims vs. the baseline's smaller feature set) accounts for the improvement.

## Potential Limitations

- **In-distribution only:** the 99.82% is on FoR for-2sec, where fake audio comes from known TTS systems (WaveNet, DeepVoice 3, etc.). Performance on newer TTS systems (ElevenLabs, Bark, XTTS) that the model hasn't seen would likely be lower.
- **2-second clips:** real-world audio is variable-length. Production deployment would need a windowing strategy (split long audio into 2-second chunks, classify each, aggregate).
- **No adversarial robustness testing:** the model hasn't been tested against audio that's been deliberately post-processed (noise injection, compression, re-recording) to evade detection.

These are acknowledged limitations, not flaws — they define the scope of what this model does and doesn't claim.

---

← [Feature Engineering](feature-engineering.md) · Back to [README](../README.md) · Next: [API Endpoint →](api.md)
