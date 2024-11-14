import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from boruta import BorutaPy

# SelectKBest
# Load the preprocessed data
df = pd.read_csv('train_processed.csv')

# Add this debugging code after loading the data
# print("\nChecking target encoded features:")
# print("dam_te unique values:", df['dam_te'].nunique())
# print("dam_te correlation with target:", df['dam_te'].corr(df['win']))
# print("\nowner_te unique values:", df['owner_te'].nunique())
# print("owner_te correlation with target:", df['owner_te'].corr(df['win']))

numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
true_numerical = []
for col in numerical_columns:
    # Exclude win, target encoded features, and binary features
    if (col != 'win' and 
        # not col.endswith('_te') and  # exclude target encoded
        not df[col].isin([0,1]).all()):
        true_numerical.append(col)
numerical_columns = [col for col in numerical_columns if col != 'win']  # exclude 'win' column

# Then separate features and target
X = df[true_numerical]  # use only numerical columns for X
print(X.columns)
y = df['win']
print(y.head())


# Initialize different feature selection methods
k_best_f = SelectKBest(score_func=f_classif, k=20)
k_best_mi = SelectKBest(score_func=mutual_info_classif, k=20)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# boruta = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=42)

# Fit and get selected features from each method
# F-score based selection
k_best_f.fit(X, y)
f_score_features = pd.DataFrame(
    {'feature': X.columns, 'f_score': k_best_f.scores_}
).sort_values('f_score', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(f_score_features['feature'], f_score_features['f_score'])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Mutual Information based selection
k_best_mi.fit(X, y)
mi_score_features = pd.DataFrame(
    {'feature': X.columns, 'mi_score': k_best_mi.scores_}
).sort_values('mi_score', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(mi_score_features['feature'], mi_score_features['mi_score'])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Random Forest importance based selection
rf.fit(X, y)
rf_importance_features = pd.DataFrame(
    {'feature': X.columns, 'importance': rf.feature_importances_}
).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
plt.bar(rf_importance_features['feature'], rf_importance_features['importance'])
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# Boruta selection
# Scale the features as Boruta works better with scaled features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# boruta.fit(X_scaled, y)
# boruta_features = pd.DataFrame({
#     'feature': X.columns,
#     'selected': boruta.support_,
#     'rank': boruta.ranking_
# }).sort_values('rank')

# Print results
print("\nTop 20 Features by F-Score:")
print(f_score_features.head(10))

print("\nTop 20 Features by Mutual Information:")
print(mi_score_features.head(10))

print("\nTop 20 Features by Random Forest Importance:")
print(rf_importance_features.head(10))

# print("\nFeatures Selected by Boruta:")
# print(boruta_features[boruta_features['selected']])

# Get features selected by at least 3 methods
top_20_f = set(f_score_features.head(5)['feature'])
top_20_mi = set(mi_score_features.head(5)['feature'])
top_20_rf = set(rf_importance_features.head(5)['feature'])
# boruta_selected = set(boruta_features[boruta_features['selected']]['feature'])

consensus_features = set.intersection(top_20_f, top_20_mi, top_20_rf)

print("\nConsensus Features (selected by all methods):")
print(consensus_features)
