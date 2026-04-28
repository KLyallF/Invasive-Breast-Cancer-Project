#!/usr/bin/env python

import pandas as pd
import numpy as np
import seaborn as sns

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import(
	classification_report, confusion_matrix, 
	roc_auc_score, roc_curve, accuracy_score,
	ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier

# -----------------------------------------------------------------------------------
# Load gene expression data into df
df = pd.read_csv("cleaned_gene_expression.csv")

# ------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------
# Random Forest classifier

# Establish parameters
rf = RandomForestClassifier(
	n_estimators=500, max_depth=5, min_samples_leaf=3,
	ccp_alpha=0.03, criterion='entropy', 
	class_weight="balanced"
)

rf.fit(X_train, Y_train)

# --------------------------------------------------------------------------------------
# Results

y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)
y_test_probs = rf.predict_proba(X_test)[:, 1]

print(
	f"Accuracy on training data = "
	f"{accuracy_score(Y_train, y_train_pred):.3f}"
)

print(
	f"Accuracy on test data = "
	f"{accuracy_score(Y_test, y_test_pred):.3f}"
)

print(
	f"Test ROC-AUC Score: "
	f"{roc_auc_score(Y_test, y_test_probs):.3f}"
)

print(
	classification_report(
		Y_test,
		y_test_pred, 
		target_names=["Ductal", "Lobular"]
	)
)

# -------------------------------------------------------------------------------------------
# Plot the ROC curve 

# Compute the false positive rate and true positive rate
# For different classification thresholds 
fpr, tpr, thresholds = roc_curve(
	Y_test, y_test_probs, pos_label=1
)

# Calculate the ROC score
roc_auc = roc_auc_score(Y_test, y_test_probs)

# Plot the ROC curve 
plt.plot(fpr, tpr, label='ROC Curve (area = %0.2f)' % roc_auc)

# ROC curve for true positive rate = false positive rate 
plt.plot([0,1], [0,1], 'k--', label='Random Classifier')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc="lower right")
plt.savefig("ROC_curve.png", dpi=300)
plt.close()

print("ROC curve saved as ROC_curve.png")

# ---------------------------------------------------------------------------------------
# Confusion matrix

cm = confusion_matrix(Y_test, y_test_pred)
disp = ConfusionMatrixDisplay(
	confusion_matrix=cm, display_labels=["Ductal", "Lobular"]
)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix: Random Forest Classifier")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()

print("Confusion matrix saved as confusion_matrix.png")

# -------------------------------------------------------------------------------------------
# Find top 10 genes for each subtype

# Find important features
importances = rf.feature_importances_
gene_names = X.columns

# Create a dataframe of top genes
importance_df = pd.DataFrame({
    'Gene': gene_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Genes by Importance:")
print(importance_df.head(10).round(4))

# ---------------------------------------------------------------------------------------
# Mean expression of top 10 genes 

# Collect names of the top 10 genes
top_10_genes = importance_df.head(10)['Gene'].tolist()

# Calculate the mean expression in ductal vs lobular
means = df.groupby(
	'CANCER_TYPE_DETAILED'
)[top_10_genes].mean().T
means.columns = ['Mean_Ductal', 'Mean_Lobular']

# Assign directionality
# If higher in lobular = associated with lobular
means['Associated_With'] = np.where(
	means['Mean_Lobular'] > means['Mean_Ductal'], 
	'Lobular', 
	'Ductal'
)

print("\nDirectional Analysis of Top 10 Features:")
print(means.round(2))

# -----------------------------------------------------------------------------------
# Make a bar chart - expression of top 10 genes 

# Complete subset needed for standard deviation calc
top_10_full = df[['CANCER_TYPE_DETAILED'] + top_10_genes]

# Prepare data for plotting
plot_df = top_10_full.melt(
	id_vars='CANCER_TYPE_DETAILED',
	value_vars=top_10_genes,
	var_name='Gene',
	value_name='Expression_Level'
)

# Plot the bar chart
sns.barplot(
	data=plot_df,
	x='Gene',
	y='Expression_Level',
	hue='CANCER_TYPE_DETAILED',
	errorbar="sd",
	capsize=0.1,
	errwidth=1
)

plt.title(
	'Expression Levels of the Top 10 Predictive Genes by Subtype'
)
plt.xlabel('Gene Symbol')
plt.ylabel('Mean Expression (Log2(RSEM + 1)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Cancer Subtype')
plt.tight_layout()

plt.savefig('Top_Genes_Bar.png', dpi=300)
plt.close()

print("Bar chart saved as Top_Genes_Bar.png")

# ------------------------------------------------------------------------------------------------
# Save model

joblib.dump(model_features, 'model_features.pkl')
print("Model features saved as model_features.pkl")

joblib.dump(rf, 'breast_cancer_rf_model.pkl')
print("Model saved as breast_cancer_rf_model.pkl")

