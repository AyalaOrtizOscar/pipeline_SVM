"""
plot_all_results.py

Genera una batería de gráficos a partir de los artefactos producidos en el pipeline:
 - modelo serializado (joblib)
 - CSV de datos finales (features...final_labeled.csv)
 - permutation_importances.csv (opcional)
 - shap_mean_abs_importance.csv (opcional)

Salida: guarda figuras PNG en el directorio --outdir (por defecto results/plots)

Uso:
 python plot_all_results.py \
   --model D:/pipeline_SVM/results/svm_final_fast/best_model.joblib \
   --data D:/pipeline_SVM/features/features_svm_harmonized_for_svm.meta_merged.final_labeled.csv \
   --outdir D:/pipeline_SVM/results/svm_final_fast/plots
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.metrics import (confusion_matrix, classification_report, precision_recall_curve,
                             roc_curve, auc, accuracy_score)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import FitFailedWarning

sns.set(style="whitegrid")

BASELINE_FEATURES = ['harmonic_percussive_ratio', 'centroid_mean', 'zcr_mean',
                     'spectral_flatness_mean', 'spectral_entropy_mean', 'onset_rate',
                     'duration_s', 'crest_factor', 'chroma_std', 'spectral_contrast_mean']

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_model(model_path):
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    return joblib.load(model_path)

def load_data(csv_path, label_col="label_fixed"):
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    # filter valid labels
    y = df[label_col].astype(str)
    mask = y.notna() & (y.str.strip()!='') & (y.str.lower()!='nan')
    X = df.loc[mask].select_dtypes(include=[np.number]).copy()
    y = y.loc[mask].copy()
    return X, y, df

def ensure_holdout(X, y, provided_holdout_dir: Path):
    """If holdout files exist (y_hold.csv, X_hold.csv) use them and reconstruct train.
    Else split with test_size=0.10 random_state=42 stratify=y.
    Returns X_train,X_hold,y_train,y_hold
    """
    y_hold_fp = provided_holdout_dir / 'y_hold.csv'
    X_hold_fp = provided_holdout_dir / 'X_hold.csv'
    if y_hold_fp.exists() and X_hold_fp.exists():
        X_hold = pd.read_csv(X_hold_fp, index_col=0)
        y_hold = pd.read_csv(y_hold_fp, index_col=0).iloc[:,0]
        if not X_hold.index.equals(y_hold.index):
            raise ValueError("Índices de X_hold y y_hold no coinciden.")
        # Ensure indices align with X; if X indices differ, try to intersect by a 'basename' column if present
        mask_hold = X.index.isin(X_hold.index)
        X_train = X.loc[~mask_hold]
        y_train = y.loc[~mask_hold]
    else:
        X_train, X_hold, y_train, y_hold = train_test_split(X, y, test_size=0.10, stratify=y, random_state=42)
    return X_train, X_hold, y_train, y_hold

def plot_confusion_matrix(y_true, y_pred, labels, out_fp):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('True')
    plt.xlabel('Pred')
    plt.title('Confusion matrix')
    plt.tight_layout()
    plt.savefig(out_fp)
    plt.close()

def plot_classification_report(y_true, y_pred, out_fp):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    df.to_csv(out_fp.with_suffix('.csv'))
    # also plot a heatmap of f1/precision/recall, excluding averages
    metrics = ['precision','recall','f1-score']
    class_rows = [idx for idx in df.index if idx not in ['accuracy', 'macro avg', 'weighted avg']]
    if class_rows:
        plt.figure(figsize=(6,4))
        sns.heatmap(df.loc[class_rows, metrics], annot=True, fmt='.2f', cmap='vlag')
        plt.title('Per-class metrics')
        plt.tight_layout()
        plt.savefig(out_fp.with_suffix('.png'))
        plt.close()

def plot_pr_roc(model, X_hold, y_hold, outdir):
    classes = list(np.unique(y_hold))
    if len(classes) < 2:
        print("No hay al menos 2 clases en holdout — se omiten ROC/PR.")
        return
    try:
        probs = model.predict_proba(X_hold)
    except Exception:
        try:
            dec = model.decision_function(X_hold)
            # convert to probabilities via softmax (approx)
            exp = np.exp(dec - np.max(dec, axis=1, keepdims=True))
            probs = exp / np.sum(exp, axis=1, keepdims=True)
            print("Advertencia: Usando aproximación softmax para probabilidades; resultados pueden no ser precisos.")
        except Exception:
            print("No probabilistic output available for PR/ROC plots")
            return
    # one-hot
    from sklearn.preprocessing import label_binarize
    y_bin = label_binarize(y_hold, classes=classes)
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1,1)
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    for i, c in enumerate(classes):
        try:
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}')
            plt.plot([0,1],[0,1],'k--')
            plt.title(f'ROC curve for class {c}')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir/f'roc_{c}.png')
            plt.close()

            precision, recall, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
            pr_auc = auc(recall, precision)
            plt.figure()
            plt.plot(recall, precision, label=f'AP={pr_auc:.3f}')
            plt.title(f'PR curve for class {c}')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir/f'pr_{c}.png')
            plt.close()
        except Exception as e:
            print(f"Error al generar ROC/PR para clase {c}: {e}")

def plot_permutation_importance(csv_fp, out_fp):
    if not csv_fp.exists():
        print('No permutation importances file found at', csv_fp)
        return
    df = pd.read_csv(csv_fp)
    # expect columns: feature, importance_mean, importance_std or similar
    col_mean = None
    for cand in ['importance_mean','mean_importance','perm_mean','importance','mean_abs_importance']:
        if cand in df.columns:
            col_mean = cand; break
    if col_mean is None:
        # fall back to second column
        if len(df.columns) >= 2:
            col_mean = df.columns[1]
        else:
            print("Formato inesperado en permutation_importances.csv")
            return
    if 'feature' not in df.columns:
        # try to infer feature column
        df = df.reset_index().rename(columns={'index':'feature'})
    df_sorted = df.sort_values(col_mean, ascending=False).head(40)
    plt.figure(figsize=(8,6))
    sns.barplot(x=col_mean, y='feature', data=df_sorted)
    plt.title('Permutation importances (top 40)')
    plt.tight_layout(); plt.savefig(out_fp); plt.close()

def plot_shap_summary(csv_fp, out_fp):
    if not csv_fp.exists():
        print('No SHAP summary CSV at', csv_fp)
        return
    df = pd.read_csv(csv_fp)
    # expect columns feature, mean_abs_shap
    if 'mean_abs_shap' in df.columns:
        df_sorted = df.sort_values('mean_abs_shap', ascending=False).head(40)
        plt.figure(figsize=(8,6))
        sns.barplot(x='mean_abs_shap', y='feature', data=df_sorted)
        plt.title('Mean |SHAP| (top 40)')
        plt.tight_layout(); plt.savefig(out_fp); plt.close()
    else:
        print('SHAP CSV missing mean_abs_shap column')

def plot_feature_distributions(df, features, label_col, outdir):
    for f in features:
        if f not in df.columns:
            continue
        try:
            plt.figure(figsize=(6,4))
            sns.kdeplot(data=df, x=f, hue=label_col, common_norm=False)
            plt.title(f'Distribution by class: {f}')
            plt.tight_layout()
            plt.savefig(outdir/f'dist_{f}.png')
            plt.close()
        except Exception as e:
            print(f"Failed distribution plot for {f}: {e}")

def plot_correlation_heatmap(df_num, top_feats, out_fp):
    try:
        if len(df_num.columns) > 30 and top_feats:
            corr = df_num[top_feats].corr()
        else:
            corr = df_num.corr()
        plt.figure(figsize=(10,8))
        sns.heatmap(corr, cmap='vlag', center=0)
        plt.title('Feature correlation')
        plt.tight_layout(); plt.savefig(out_fp); plt.close()
    except Exception as e:
        print("Failed correlation heatmap:", e)

def plot_pca_scatter(X, y, out_fp, n_components=2, top_feats=None):
    from sklearn.decomposition import PCA
    try:
        if top_feats:
            Xp = X[top_feats].fillna(X[top_feats].mean())
        else:
            Xp = X.fillna(X.mean())
        pca = PCA(n_components=n_components)
        proj = pca.fit_transform(Xp)
        plt.figure(figsize=(6,5))
        dfp = pd.DataFrame({'pc1':proj[:,0],'pc2':proj[:,1],'label':y.values})
        sns.scatterplot(data=dfp, x='pc1', y='pc2', hue='label', alpha=0.7)
        plt.title('PCA scatter (2D)')
        plt.tight_layout(); plt.savefig(out_fp); plt.close()
    except Exception as e:
        print("PCA scatter failed:", e)

def plot_pairplot_baseline(df, baseline_features, label_col, out_fp, max_samples=500):
    cols = [c for c in baseline_features if c in df.columns]
    if len(cols) < 2:
        print('Not enough baseline features for pairplot')
        return
    try:
        # sample to avoid huge pairplots
        if len(df) > max_samples:
            df_sample = df[cols + [label_col]].dropna().sample(max_samples, random_state=42)
        else:
            df_sample = df[cols + [label_col]].dropna()
        sns.pairplot(df_sample, hue=label_col, plot_kws={'alpha':0.6}, corner=True)
        plt.savefig(out_fp)
        plt.close()
    except Exception as e:
        print('pairplot failed:', e)
def plot_learning_curve(model, X, y, out_fp):
    """
    Versión robusta de learning_curve:
    - calcula train_sizes factibles donde cada clase tendría >=2 ejemplos en el train subset
    - usa StratifiedKFold con n_splits adaptativo (<=3, <=min_count_por_clase)
    - omite la curva si no hay train_sizes factibles
    """
    try:
        # si es CalibratedClassifierCV prefit, usar el estimador base
        if isinstance(model, CalibratedClassifierCV) and getattr(model, 'cv', None) == 'prefit':
            base_model = model.estimator
        else:
            base_model = model

        # asegurar pandas Series para y
        y_ser = pd.Series(y).reset_index(drop=True)
        n_samples = len(y_ser)
        if n_samples < 10:
            print(f"Skipped learning curve: muy pocos samples ({n_samples})")
            return

        class_counts = y_ser.value_counts()
        if class_counts.empty:
            print("Skipped learning curve: y vacío")
            return
        min_class = int(class_counts.min())

        # candidate train fractions (empieza en 0.2)
        candidate_fracs = np.linspace(0.2, 1.0, 5)

        # seleccionar fracciones donde para cada clase floor(frac * count_class) >= 2
        feasible_fracs = []
        for f in candidate_fracs:
            enough = all((np.floor(f * class_counts.values) >= 2))
            # adicionalmente asegurar que el total de training samples sea >= sum(2 per class)
            if enough and int(np.floor(f * n_samples)) >= (2 * len(class_counts)):
                feasible_fracs.append(f)

        if not feasible_fracs:
            print(f"Skipped learning curve: no existen train_sizes factibles (min_class={min_class}, total={n_samples})")
            return

        train_sizes = np.array(feasible_fracs)

        # decidir n_splits: no más que 3 y no mayor que min_class
        n_splits = min(3, min_class) if min_class >= 2 else 0
        if n_splits < 2:
            print(f"Skipped learning curve: clases insuficientes para CV (min samples per class={min_class})")
            return

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # usar n_jobs=1 en Windows para evitar oversubscription; suprimir FitFailedWarning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FitFailedWarning)
            train_sizes_abs, train_scores, val_scores = learning_curve(
                base_model, X, y_ser, cv=cv, n_jobs=1,
                train_sizes=train_sizes, scoring='f1_macro', error_score=np.nan
            )

        # si todo salió nan, omitir
        if train_scores.size == 0 or np.all(np.isnan(val_scores)):
            print("Learning curve returned only NaNs; omitiendo gráfico.")
            return

        train_mean = np.nanmean(train_scores, axis=1)
        val_mean = np.nanmean(val_scores, axis=1)

        plt.figure()
        plt.plot(train_sizes_abs, train_mean, 'o-', label='train')
        plt.plot(train_sizes_abs, val_mean, 'o-', label='cv')
        plt.xlabel('Training samples')
        plt.ylabel('F1 macro')
        plt.legend()
        plt.title('Learning curve')
        plt.tight_layout()
        plt.savefig(out_fp)
        plt.close()

    except Exception as e:
        print('Failed to compute learning curve:', e)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--outdir', default=None)
    p.add_argument('--label-col', default='label_fixed')
    p.add_argument('--baseline-features', nargs='+', default=BASELINE_FEATURES)
    p.add_argument('--skip-learning-curve', action='store_true', help='Omitir cálculo de learning curve (útil para runs rápidos)')
    args = p.parse_args()

    model_path = Path(args.model)
    csv_path = Path(args.data)
    outdir = Path(args.outdir) if args.outdir else model_path.parent / 'plots'
    safe_mkdir(outdir)

    print('Loading model...')
    model = load_model(model_path)
    if not hasattr(model, 'predict'):
        raise ValueError("Modelo inválido: no tiene 'predict'.")

    print('Loading data...')
    X, y, df_full = load_data(csv_path, label_col=args.label_col)

    print('Preparing holdout...')
    X_train, X_hold, y_train, y_hold = ensure_holdout(X, y, outdir)

    print('Predicting holdout...')
    try:
        y_pred = model.predict(X_hold)
    except Exception as e:
        print('Model prediction failed on holdout:', e)
        return

    labels = list(np.unique(y_hold))
    plot_confusion_matrix(y_hold, y_pred, labels, outdir/'confusion_matrix.png')
    plot_classification_report(y_hold, y_pred, outdir/'classification_report')
    plot_pr_roc(model, X_hold, y_hold, outdir)

    # permutation importances
    plot_permutation_importance(model_path.parent/'permutation_importances.csv', outdir/'permutation_importances.png')
    # shap
    plot_shap_summary(model_path.parent/'shap_mean_abs_importance.csv', outdir/'shap_mean_abs.png')

    # extract top_feats if possible (soportar pipelines y CalibratedClassifierCV)
    top_feats = args.baseline_features
    try:
        candidate = getattr(model, 'estimator', model)
        selector = None
        if hasattr(candidate, 'named_steps'):
            selector = candidate.named_steps.get('selector', None)
        elif hasattr(candidate, 'get_params'):
            # buscar paso 'selector' en pipeline-like params
            try:
                # Si pipeline está anidado, intentar acceder por atributo
                selector = candidate.get_params().get('selector', None)
            except Exception:
                selector = None
        if selector is not None and hasattr(selector, 'get_support'):
            # selector puede ser boolean mask o similar
            mask = selector.get_support()
            top_feats = list(X.columns[mask])
    except Exception:
        pass

    # distributions for top selected features
    print('Plotting distributions...')
    plot_feature_distributions(df_full, top_feats, args.label_col, outdir)

    # correlation
    plot_correlation_heatmap(X, top_feats, outdir/'correlation_heatmap.png')

    # PCA scatter
    try:
        plot_pca_scatter(X, y.loc[X.index], outdir/'pca_baseline.png', top_feats=top_feats)
    except Exception as e:
        print('PCA baseline failed:', e)

    # pairplot baseline
    try:
        plot_pairplot_baseline(df_full, args.baseline_features, args.label_col, outdir/'pairplot_baseline.png')
    except Exception as e:
        print('pairplot failed:', e)

    # learning curve (use pipeline without heavy cv)
    if not args.skip_learning_curve:
        try:
            plot_learning_curve(model, X, y, outdir/'learning_curve.png')
        except Exception as e:
            print('learning curve error:', e)
    else:
        print("Omitiendo learning curve por --skip-learning-curve")

    print(f'Todos los gráficos generados (los que fueron posibles). Directorio: {outdir}')

if __name__ == '__main__':
    main()
