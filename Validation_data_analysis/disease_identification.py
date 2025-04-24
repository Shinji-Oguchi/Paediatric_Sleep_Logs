#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Using Kizugawa full dataset (ages 3–5) to identify developmental disorders, atopy, and asthma,
and to compare different feature sets and visualize model performance and feature importance.

Author: Shinji Oguchi
Date: 2025-4-25
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import shap

# ----------------------------------------------------------------------------
# 1) Load and filter Kizugawa data
# ----------------------------------------------------------------------------
data_file = "kizugawa_full_data_age3to5_2024024.csv"
kiz_df = pd.read_csv(data_file, index_col=0)
# keep rows where 'devdis' and 'sex' are not missing
filtered = kiz_df[kiz_df['devdis'].notna() & kiz_df['sex'].notna()]  

# ----------------------------------------------------------------------------
# 2) Prepare feature matrix and targets for developmental disorder, atopy,and asthma
# ----------------------------------------------------------------------------
features = [
    'sex', 'age',
    'Circadian.1', 'Circadian.2', 'Circadian.3', 'Circadian.4',
    'Gap.1', 'Gap.2', 'Gap.3', 'Gap.4',
    'Error.1', 'Error.2', 'Error.3'
]
X = filtered[features]
y_devdis = filtered['devdis']
y_atopy  = filtered['atopy']
y_asthma = filtered['asthma']

# ----------------------------------------------------------------------------
# 3) Utility function: train logistic model, plot ROC
# ----------------------------------------------------------------------------
def train_plot_roc(X, y, label, color):
    # split, scale, fit
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=1234
    )
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std  = scaler.transform(X_test)
    lr = LogisticRegression(
        max_iter=100000, penalty='elasticnet', solver='saga',
        l1_ratio=0.5, class_weight='balanced', random_state=1234
    )
    lr.fit(X_train_std, y_train)

    # ROC curve
    probs = lr.predict_proba(X_test_std)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, label=f'{label} (AUC={roc_auc:.3f})')
    return lr, scaler

# ----------------------------------------------------------------------------
# 4) Plot ROC for each condition on same axes
# ----------------------------------------------------------------------------
plt.figure(figsize=(5, 4.5))
train_plot_roc(X, y_devdis, 'Developmental disorder', 'red')
train_plot_roc(X, y_atopy,  'Atopy',                'blue')
train_plot_roc(X, y_asthma, 'Asthma',               'green')
# random/ideal lines
plt.plot([0,1], [0,1], 'k--', label='Random')
plt.plot([0,0,1], [0,1,1], 'k-.', label='Ideal')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves by Condition')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 5) Compare feature sets for developmental disorders
# ----------------------------------------------------------------------------
feature_sets = {
    'Distance':    ['sex', 'age', 'Distance'],
    'Quartile':    ['sex', 'age', 'quartile'],
    'Sleep score': ['sex', 'age', 'sleep_score'],
    'AgeSex only': ['sex', 'age']
}

y = y_devdis
plt.figure(figsize=(5, 4.5))
for label, cols in feature_sets.items():
    train_plot_roc(filtered[cols], y, label, color=None)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DevDis Prediction: Different Feature Sets')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 6) Final model with combined features + Distance
# ----------------------------------------------------------------------------
combined_feats = features + ['Distance']
X_comb = filtered[combined_feats]
plt.figure(figsize=(5, 4.5))
lr_final, scaler_final = train_plot_roc(
    X_comb, y_devdis, 'Combined + Distance', 'magenta'
)
plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DevDis Prediction: Full Feature Model')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# 7) SHAP analysis for final model
# ----------------------------------------------------------------------------
# explain on test set
X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_comb, y_devdis, test_size=0.4, stratify=y_devdis, random_state=1234
)
X_train_std_f = scaler_final.transform(X_train_f)
X_test_std_f  = scaler_final.transform(X_test_f)
explainer = shap.Explainer(lr_final, X_train_std_f, feature_names=combined_feats)
shap_values = explainer(X_test_std_f)

plt.figure(figsize=(6,4))
shap.plots.beeswarm(shap_values)
plt.title('SHAP Beeswarm: Full Feature Model')
plt.tight_layout()
plt.show()
