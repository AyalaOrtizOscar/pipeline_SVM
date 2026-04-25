# 2) matriz de confusión y reporte holdout (si guardaste X_hold,y_hold, sino vuelve a hacerlo)
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# recalcula X_hold,y_hold (mismo filtrado/stratify que en entrenamiento)
df = pd.read_csv("D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv", low_memory=False)
label_col="label_fixed"
X = df.select_dtypes(include=[np.number])
y = df[label_col].astype(str)
mask = y.notna() & (y.str.strip()!='') & (y.str.lower()!='nan')
X = X.loc[mask]; y = y.loc[mask]

from sklearn.model_selection import train_test_split
X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)

y_pred = model.predict(X_hold)
print(classification_report(y_hold, y_pred, zero_division=0))

cm = confusion_matrix(y_hold, y_pred, labels=model.classes_)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("pred")
plt.ylabel("true")
plt.show()


