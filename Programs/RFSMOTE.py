#!/usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import(
        classification_report, roc_auc_score
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# -----------------------------------------------------------------------------------
# Load gene expression data into df
df = pd.read_csv("cleaned_gene_expression.csv")

# Define X (features) and y (target variable)
X = df.drop(
        columns=["Patient", "CANCER_TYPE_DETAILED", "Subtype_Label"]
)
Y = df["Subtype_Label"]

# Convert column names to strings
X.columns = X.columns.astype(str)

# Save the model features
model_features = X.columns.tolist()

# Split Data
X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=1
)

# ------------------------------------------------------------------------------------
# Apply SMOTE

# Apply SMOTE to training data only
sm = SMOTE(random_state=1)

# Creates synthetic lobular samples
X_train_sm, Y_train_sm = sm.fit_resample(X_train, Y_train)

# ----------------------------------------------------------------------------------------
# Random Forest classifier with SMOTE

# Establish parameters
# Remove class_weight="balanced"
rf_smote = RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_leaf=3,
        ccp_alpha=0.03, criterion='entropy'
)

rf_smote.fit(X_train_sm, Y_train_sm)

# ------------------------------------------------------------------------------------------------
# Random forest classifier without SMOTE

# Establish parameters
rf = RandomForestClassifier(
	n_estimators=500, max_depth=5, min_samples_leaf=3,
	ccp_alpha=0.03, criterion='entropy',
	class_weight="balanced"
)

rf.fit(X_train, Y_train)

# --------------------------------------------------------------------------------------
# Results (SMOTE)

y_smote_pred = rf_smote.predict(X_test)
y_smote_probs = rf_smote.predict_proba(X_test)[:, 1]

print("\nClassification Report (RF with SMOTE):")
print(
	classification_report(
		Y_test,
		y_smote_pred,
		target_names=["Ductal", "Lobular"]
	)
)

print(f"ROC-AUC: {roc_auc_score(Y_test, y_smote_probs):.3f}")

# -----------------------------------------------------------------------------------------------
# Results (without SMOTE)

y_test_pred = rf.predict(X_test)
y_test_probs = rf.predict_proba(X_test)[:, 1]

print("\nClassification Report (RF without SMOTE):")
print(
	classification_report(
		Y_test,
		y_test_pred,
		target_names=["Ductal", "Lobular"]
	)
)
print(f"ROC-AUC: {roc_auc_score(Y_test, y_test_probs):.3f}")
