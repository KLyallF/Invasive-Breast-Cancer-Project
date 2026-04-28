#!/usr/bin/env python

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report

# --------------------------------------------------------------------
# Load the model and the features list

rf = joblib.load('breast_cancer_rf_model.pkl')
features = joblib.load('model_features.pkl')

# -------------------------------------------------------------------
# Load and clean the data

# Load the data sets
model_info = pd.read_csv("Model.csv")
expr_data = pd.read_csv(
	"OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
).set_index("ModelID")

# Align the gene names with the hugo symbols in the model
expr_data.columns = expr_data.columns.str.split(' ').str[0]

# Keep only the breast cancer cell lines
breast_ids = model_info.query(
	"OncotreeLineage == 'Breast'"
)["ModelID"]

# Order the genes in the same way as model data set
X = (expr_data
	.loc[expr_data.index.intersection(breast_ids)]
	.reindex(columns=features, fill_value=0))

print("Data loaded and cleaned.")

# ----------------------------------------------------------------------------------------------
# Cell line subtype predictions 

# Use RF model to make predictions on subtype of cell lines
probs = rf.predict_proba(X)

# Create a results dataframe 
results = pd.DataFrame({
	'ModelID': X.index,
	'Prob_Ductal': probs[:, 0],
	'Prob_Lobular': probs[:, 1],
	# Relative Ranking: if lob prob > duct prob predict lobular
		#else = ductal
	'Pred': np.where(
	probs[:, 1] > probs[:, 0], "Lobular", "Ductal"
)})

# Merge the dataframe with cell line info
info_cols = ['ModelID', 'CellLineName', 'OncotreeSubtype']
results = results.merge(model_info[info_cols], on='ModelID')

# Extract known labels for the representative lines lists
results['Known Subtype'] = results['OncotreeSubtype']

# Extract known labels for ductal and lobular subtypes
results['Known Duct or Lob'] = results['OncotreeSubtype'].str.extract(
	r'(Ductal|Lobular)', expand=False, flags=2
)
 
print("Cell line predictions complete.")

# --------------------------------------------------------------------------------------------------
# Identify top representative lines 

# Select columns to show
cols_D = ['CellLineName', 'Prob_Ductal', 'Known Subtype']
cols_L = ['CellLineName', 'Prob_Lobular', 'Known Subtype']

# Sort the results
sorted_duc = results.sort_values('Prob_Ductal', ascending=False)
sorted_lob = results.sort_values('Prob_Lobular', ascending=False)

# Round the probability columns 
sorted_duc['Prob_Ductal'] = sorted_duc['Prob_Ductal'].round(4)
sorted_lob['Prob_Lobular'] = sorted_lob['Prob_Lobular'].round(4)

# Top 5 representative lines
top5_duc = sorted_duc.head(5)
top5_lob = sorted_lob.head(5)

print("\n Top 5 Most Representative Ductal Cell Lines")
print(top5_duc[cols_D])

print("\n Top 5 Most Representative Lobular Cell Lines")
print(top5_lob[cols_L])

# ----------------------------------------------------------------------------
# Classifcation report 

val_data = results.dropna(subset=['Known Duct or Lob'])
print(
	classification_report(
	val_data['Known Duct or Lob'], val_data['Pred']
))

# --------------------------------------------------------------------------
# CDH1 validation boxplot 

plt.figure(figsize=(6, 6))

# Select ModelIDs of the most representative lines
top5_duc = sorted_duc.head(5)['ModelID']
top5_lob = sorted_lob.head(5)['ModelID']

# Extract CDH1 gene expression values for these ModelIDs 
sns.boxplot(data=[expr_data.loc[top5_duc, 'CDH1'],
	expr_data.loc[top5_lob, 'CDH1']],
	width=0.4
)

# Add horizontal gridlines
plt.grid(axis='y', alpha=0.7)

# Add tick marks on y axis 
plt.yticks(np.arange(0, 11, 1), fontsize=14)

# Add labels 
plt.xticks([0, 1], ['Ductal', 'Lobular'], fontsize=14)
plt.ylabel(
	"CDH1 Expression (Log2(TPM + 1))", 
	labelpad=15, fontsize=14)
plt.xlabel(
	"Model Cell Line Predictions",
	labelpad=15, fontsize=14)

# Show full range of data 
plt.ylim(-0.5, 10.5)

plt.title("CDH1 Expression Levels of the Predicted Ductal\n"
	" and Lobular Cell Lines", fontsize=16, pad=30)
plt.tight_layout(pad=3.0)

# Improve spacing
plt.subplots_adjust(left=0.15, right=0.95, top=0.85, bottom=0.15)

plt.savefig('CDH1_Plot.png', 
	dpi=300, 
	bbox_inches='tight',
	pad_inches=0.2
)
plt.close()

print("CDH1 plot saved as CDH1_Plot.png")

# ------------------------------------------------------------------------------------------------

# Extract top 5 with their model IDs 
top5_ducM = sorted_duc.head(5)['ModelID']
top5_lobM = sorted_lob.head(5)['ModelID'] 

# Find the Top 10 genes the model is using 
#importances = rf.feature_importances_
#feat_imp = pd.DataFrame({
#    'Gene': features, 
#    'Importance': importances
#}).sort_values('Importance', ascending=False).head(10)

#top_genes = feat_imp['Gene'].tolist()

# Compare expression of these 10 genes in the cell lines
#duc_expr = X.loc[top5_ducM, top_genes].mean()
#lob_expr = X.loc[top5_lobM, top_genes].mean()

#comparison = pd.DataFrame({
#    'Importance': feat_imp.set_index('Gene')['Importance'],
#    'Mean_Top_Ductal': duc_expr,
#    'Mean_Top_Lobular': lob_expr
#})

# Calculate the difference in expression
	# To see which genes distinguish the cell lines
#comparison['Diff'] = comparison['Mean_Top_Ductal'] - \
#	comparison['Mean_Top_Lobular']

#print("\nTop 10 Model Genes: Expression in Predicted Cell Lines")
#print(comparison.sort_values('Importance', ascending=False))

# Find the top 10 genes from the model
importances = rf.feature_importances_
feat_imp = pd.DataFrame({
	'Gene': features,
	'Importance': importances
}).sort_values('Importance', ascending=False).head(10)

top_genes = feat_imp['Gene'].tolist()

# Calculate mean gene expression
means_df = pd.DataFrame({
	'Mean_Ductal': X.loc[top5_ducM, top_genes].mean(),
	'Mean_Lobular': X.loc[top5_lobM, top_genes].mean()
})

# Assign directionality
means_df['Associated_With'] = np.where(
	means_df['Mean_Lobular'] > means_df['Mean_Ductal'],
	'Lobular',
	'Ductal'
)

print("\nDirectional Analysis of Top 10 Features in the Cell Lines:")
print(means_df.round(2))

# ------------------------------------------------------------------
# Create a heatmap of the top 20 genes

# Find top 20 most important genes
top_10_genes = (
    pd.DataFrame({'Gene': features, 'Importance': importances})
    .sort_values('Importance', ascending=False)
    .head(10)['Gene']
)

# Combine top Lobular + Ductal IDs (keep order)
combined_ids = pd.concat([top5_ducM, top5_lobM])

# Extract expression data
heatmap_data = X.loc[combined_ids, top_10_genes]

# Create row labels (CellLineName + subtype)
# Create a mapping - provide an ID, get back cell line name
id_to_name = results.set_index('ModelID')['CellLineName']

# Select the cell line names
row_names = id_to_name.loc[combined_ids]

# Group subtypes and cell lines
group_labels = ['Ductal'] * len(top5_ducM) \
	+ ['Lobular'] * len(top5_lobM)

# Pair up the cell line names and subtype 
row_labels = [
    f"{name} ({group})"
    for name, group in zip(row_names, group_labels)
]

# Plot heatmap
plt.figure(figsize=(12, 8))

ax = sns.heatmap(
    heatmap_data,
    cmap='viridis',
    yticklabels=row_labels,
    cbar_kws={'label': 'Expression'}
)

# Change colour bar font size
cbar = ax.collections[0].colorbar 
cbar.ax.tick_params(labelsize=14)
cbar.set_label('Expression', size=14, labelpad=15)

# Rotate x-axis labels
plt.xticks(rotation=45, ha='right', fontsize=14)

# Y-axis labels
plt.yticks(fontsize=14)

# Add a separator between Lobular and Ductal
plt.axhline(len(top5_ducM), color='white', linewidth=10)

# Title
plt.title(
	"Expression of the Top 10 Predictive Genes in the " 
	"Representative Cell Lines",
	fontsize=17,
	pad=20
)

# Axis labels
plt.xlabel("Gene", fontsize=14, labelpad=20)
plt.ylabel(
	"Cell Line (Predicted Subtype)", 
	fontsize=14,
	labelpad=20
)

plt.tight_layout()
plt.savefig('Gene_Heatmap.png', dpi=300)
plt.close()

print("Gene heatmap saved as Gene_Heatmap.png")
