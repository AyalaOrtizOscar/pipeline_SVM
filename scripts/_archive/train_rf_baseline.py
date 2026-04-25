# train_rf_baseline.py
import pandas as pd, numpy as np, joblib, json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
ROOT = Path("D:/pipeline_SVM")
F = ROOT/"features"/"features_svm_baseline.csv"
OUT = ROOT/"results"/"rf_run"; OUT.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(F, low_memory=False)
label = 'label_clean' if 'label_clean' in df.columns else 'label'
meta = ['filepath','label','label_clean','mic_type','experiment','duration','duration_s']
X = df.select_dtypes(include=[np.number])
y = df[label].astype(str)
le = LabelEncoder(); y_enc = le.fit_transform(y)
g = df['experiment'] if 'experiment' in df.columns else (df['mic_type'] if 'mic_type' in df.columns else np.arange(len(df)))
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y_enc, g))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y_enc[train_idx], y_enc[test_idx]
pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler()),("rf", RandomForestClassifier(n_estimators=200, max_depth=40, class_weight='balanced', random_state=42, n_jobs=-1))])
pipe.fit(X_train, y_train)
joblib.dump(pipe, OUT/"rf_baseline.joblib")
pred = pipe.predict(X_test)
pred_labels = le.inverse_transform(pred); true_labels = le.inverse_transform(y_test)
pd.DataFrame(classification_report(true_labels, pred_labels, output_dict=True)).transpose().to_csv(OUT/"classification_report_holdout.csv")
pd.DataFrame(confusion_matrix(true_labels,pred_labels), index=le.classes_, columns=le.classes_).to_csv(OUT/"confusion_matrix_holdout.csv")
# feature importances
fi = pipe.named_steps['rf'].feature_importances_
pd.DataFrame({"feature":X.columns,"importance":fi}).sort_values("importance",ascending=False).to_csv(OUT/"feature_importances_rf.csv", index=False)
print("RF done. Results:", OUT)
