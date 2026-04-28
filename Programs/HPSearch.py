#!/usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import(
	classification_report, accuracy_score,
	f1_score, roc_auc_score
)

# -----------------------------------------------------------------------------
# Load and prep the data

# Load clean gene expression data into df
df = pd.read_csv("cleaned_gene_expression.csv")

# Define X (features) and y (target variable)
X = df.drop(
	columns=["Patient", "CANCER_TYPE_DETAILED", "Subtype_Label"]
)
Y = df["Subtype_Label"]

# Convert column names to strings 
X.columns = X.columns.astype(str)

# --------------------------------------------------------------------------------------
# Training Random Forest 

# Split into train and test data 
X_train, X_test, Y_train, Y_test = train_test_split(
	X, Y, test_size=0.2, random_state=1)

# ----------------------------------------------------------------------------------
# Hyperparameter search

RF_model = RandomForestClassifier(
	random_state=1,
	class_weight="balanced")

# Parameters for gridsearch
params = {
	"n_estimators": [400, 500, 600],
	"max_depth": [4, 5, 6],
	"criterion": ["entropy", "gini"],
	"ccp_alpha": [0.02, 0.03, 0.04],
	"min_samples_leaf": [3, 4, 5]
}

# Conduct the grid search
rf_cv = GridSearchCV(
	RF_model,
	params,
	scoring={"accuracy": "accuracy", "f1": "f1"},
	refit="f1",
	return_train_score=True,
	cv=5,
	verbose=3,
	n_jobs=-1
)

# Fit the models with these parameters
rf_cv = rf_cv.fit(X_train, Y_train)

# -----------------------------------------------------------------------------------------------------------
# Results 

# Find the best parameters
best_params = rf_cv.best_params_

# Store results as a data frame
cv_results = pd.DataFrame(rf_cv.cv_results_)

# Pick out the f1 score columns 
cols_to_show = [
	col for col in cv_results.columns
	if "param_" in col
	or "mean_test_f1" in col 
	or "mean_train_f1" in col
]

# Print the top combinations
print("\nTop Parameter Combinations (Sorted by Test F1):")
print(
	cv_results[cols_to_show]
	.sort_values("mean_test_f1", ascending=False)
	.head(5)
)

# Find the best model
best_model = rf_cv.best_estimator_

# Final evaluation on test set
y_final_pred = best_model.predict(X_test)

# Print test report 
print("\nFinal Test Set Report:")
print(f"Best Params: {rf_cv.best_params_}")
print(classification_report(Y_test, y_final_pred))

# ------------------------------------------------------------------------------------------
# Try a different threshold on the best model 

# Use the best model from search 
y_probs = best_model.predict_proba(X_test)[:, 1]

# Change threshold
threshold = 0.40
y_pred_high_precision = (y_probs >= threshold).astype(int)

# Print classification report 
print("\nClassification Report with Threshold = 0.40") 
print(classification_report(Y_test, y_pred_high_precision))
