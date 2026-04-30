# === IMPORTS ===
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE

# === LOAD DATA ===
df = pd.read_csv(r"C:\Users\Darshan\Desktop\Hypertension Prediction\Hypertension-risk-model-main.csv")

# === IMPUTE MISSING VALUES ===
imputer = KNNImputer(n_neighbors=3)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Round categorical columns
for col in ['male', 'currentSmoker', 'BPMeds', 'diabetes', 'Risk']:
    df_imputed[col] = df_imputed[col].round().astype(int)

# === FEATURE ENGINEERING ===
df_imputed['pulse_pressure'] = df_imputed['sysBP'] - df_imputed['diaBP']


# Save processed dataset WITHOUT bmi_category & age_group
processed_path = r"C:\Users\Darshan\Desktop\Hypertension Prediction\NEW_HyperTensionCode\processed_dataset.csv"
df_imputed.to_csv(processed_path, index=False)

# === SPLIT FEATURES & TARGET ===
X = df_imputed.drop('Risk', axis=1)
y = df_imputed['Risk']

# === TRAIN/TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === SCALE ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "C:/Users/Darshan/Desktop/Hypertension Prediction/NEW_HyperTensionCode/Model/scaler.pkl")

# === FEATURE SELECTION ===
manual_features = ['male', 'cigsPerDay', 'BPMeds', 'diabetes', 'pulse_pressure']
selector = SelectKBest(score_func=f_classif, k=10)
selector.fit(X_train_scaled, y_train)
auto_features = X.columns[selector.get_support()].tolist()

# Combine
hybrid_features = sorted(list(set(manual_features + auto_features)))
print(" Manually selected features:", manual_features)
print(" K-Best selected features:", auto_features)
print(" Final hybrid feature set:", hybrid_features)

# Save feature list
joblib.dump(hybrid_features, "C:/Users/Darshan/Desktop/Hypertension Prediction/NEW_HyperTensionCode/Model/feature_columns.pkl")

# Subset data
X_train_selected = X_train[hybrid_features]
X_test_selected = X_test[hybrid_features]

# === HANDLE IMBALANCE WITH SMOTE ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

# === SCALE AFTER FEATURE SELECTION ===
scaler_final = StandardScaler()
X_train_final = scaler_final.fit_transform(X_train_resampled)
X_test_final = scaler_final.transform(X_test_selected)
joblib.dump(scaler_final, "C:/Users/Darshan/Desktop/Hypertension Prediction/NEW_HyperTensionCode/Model/final_scaler.pkl")

# === TRAIN MULTIPLE MODELS ===
models = {
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42, class_weight='balanced'),
    'Decision Tree': DecisionTreeClassifier(random_state=42, class_weight='balanced')
}

accuracy_results = {}
for name, model in models.items():
    model.fit(X_train_final, y_train_resampled)
    y_pred = model.predict(X_test_final)
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = acc
    print(f"\n {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

# === SAVE BEST MODEL ===
best_model_name = max(accuracy_results, key=accuracy_results.get)
best_model = models[best_model_name]
joblib.dump(best_model, "C:/Users/Darshan/Desktop/Hypertension Prediction/NEW_HyperTensionCode/Model/hypertension_model.pkl")
print(f"\n Best Model Saved: {best_model_name}")


from imblearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# === CROSS-VALIDATION WITH SMOTE INSIDE FOLDS ===

# Pipeline: SMOTE -> Scaling -> Random Forest
rf_pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
])

# Perform 10-fold cross-validation
cv_scores = cross_val_score(
    rf_pipeline,
    X[hybrid_features],  # use same final features
    y,
    cv=10,
    scoring='accuracy'
)

print("\n=== Cross-Validation Results (Random Forest with SMOTE inside folds) ===")
print(f"CV Accuracy Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean():.4f}")
print(f"Std Deviation: {cv_scores.std():.4f}")

# Save feature importance for Random Forest
if best_model_name == "Random Forest":
    importance = pd.Series(best_model.feature_importances_, index=hybrid_features)
    importance.sort_values(ascending=False).to_csv("C:/Users/Darshan/Desktop/Hypertension Prediction/NEW_HyperTensionCode/Model/feature_importance.csv")


# === CONFUSION MATRIX FOR BEST MODEL ===
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict again with best model to get confusion matrix
y_pred_best = best_model.predict(X_test_final)
cm = confusion_matrix(y_test, y_pred_best)

# Plot confusion matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()

# Save confusion matrix as image
conf_matrix_path = "C:/Users/Darshan/Desktop/Hypertension Prediction/NEW_HyperTensionCode/Model/confusion_matrix.png"
plt.savefig(conf_matrix_path)
plt.show()