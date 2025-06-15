# Traffic Congestion Predictor

# Import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv("dataset.csv")
df.columns = df.columns.str.strip()  # Clean any accidental whitespaces

print("Dataset Loaded Successfully")
print(df.head())

# Drop non-informative columns
df.drop(columns=['Day', 'Date'], inplace=True)

# Convert 'Weather' to string for encoding
df['Weather'] = df['Weather'].astype(str)

# One-hot encode categorical feature 'Weather'
df = pd.get_dummies(df, columns=['Weather'], drop_first=True)

# Convert boolean columns (from get_dummies) to int
df = df.astype({col: int for col in df.columns if df[col].dtype == bool})

# Create synthetic 'Hour' feature from Zone
df['Hour'] = df['Zone'] % 24

# Convert 5 traffic levels into 3 broader classes
# 1–2 = Low, 3 = Medium, 4–5 = High
def bin_traffic(x):
    if x <= 2:
        return 1  # Low
    elif x == 3:
        return 2  # Medium
    else:
        return 3  # High

df['Traffic_Level'] = df['Traffic'].apply(bin_traffic)
print(df.head())

# Separate features and target
X = df.drop('Traffic', axis=1)
y = df['Traffic_Level']

# Stratified split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier with basic tuning
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
clf.fit(X_train_scaled, y_train)

# Predict
y_pred = clf.predict(X_test_scaled)

# Accuracy & Classification Report
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", linewidths=1, linecolor='gray')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Feature Importance Plot
importances = clf.feature_importances_
features = X.columns

imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
imp_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=imp_df, palette='crest')
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.show()

# Save predictions with actuals to CSV
result_df = X_test.copy()
result_df['Actual_Traffic'] = y_test.values
result_df['Predicted_Traffic'] = y_pred
result_df.to_csv("traffic_predictions_output.csv", index=False)
print("Predictions saved to 'traffic_predictions_output.csv'")

# Count of Predicted Traffic Levels (1, 2, 3)
plt.figure(figsize=(6, 4))
sns.countplot(x=y_pred, palette='Set2')

plt.title("Predicted Traffic Levels Distribution")
plt.xlabel("Traffic Level (1 = Low, 2 = Medium, 3 = High)")
plt.ylabel("Number of Predictions")
plt.ylim(0, 160)  # adjustable
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Average Predicted Traffic Level for Top 10 Zones (sorted by traffic)
top_zones = result_df.groupby('Zone')['Predicted_Traffic'].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_zones.index.astype(str), y=top_zones.values, palette='coolwarm')

plt.title("Top 10 Zones by Avg Predicted Traffic Level")
plt.xlabel("Zone")
plt.ylabel("Average Traffic Level")
plt.ylim(1, 3.2)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# Predicted Traffic vs Temperature 
sample_df = result_df.sample(200, random_state=42)

plt.figure(figsize=(8, 5))
sns.scatterplot(x='Temperature', y='Predicted_Traffic', data=sample_df)

plt.title("Predicted Traffic vs Temperature (Sample of 200)")
plt.xlabel("Temperature")
plt.ylabel("Predicted Traffic Level")
plt.ylim(0.5, 3.5)
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()

# Confusion matrix with labels
cm = confusion_matrix(y_test, y_pred)
labels = ['Low (1)', 'Med (2)', 'High (3)']

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', xticklabels=labels, yticklabels=labels)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Feature Importance
feat_importances = pd.Series(clf.feature_importances_, index=X.columns)
top_features = feat_importances.sort_values(ascending=False).head(8)

plt.figure(figsize=(9, 5))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")

plt.title("Top 8 Feature Importances")
plt.xlabel("Importance Score")
plt.grid(axis='x', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()