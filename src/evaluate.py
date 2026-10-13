"""Educational breast cancer Decision Tree comparison. Not for medical use."""
from pathlib import Path
import argparse
import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, balanced_accuracy_score, recall_score, f1_score
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

FEATURES = ['concave points_mean', 'radius_worst', 'perimeter_worst', 'area_worst', 'texture_worst', 'concave points_worst', 'perimeter_mean', 'area_mean', 'radius_mean', 'concavity_mean']

def evaluate(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    missing = set(FEATURES + ['diagnosis']) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    X = df[FEATURES].copy()
    y = df['diagnosis'].map({'B': 0, 'M': 1})
    if y.isna().any() or X.isna().any().any():
        raise ValueError('Missing or invalid data')
    xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = {'tree__criterion': ['gini', 'entropy'], 'tree__max_depth': [1, 2, 3, 4, 5, None], 'tree__min_samples_leaf': [1, 2, 4, 8]}
    results = {}
    for key, use_smote in [('with_smote', True), ('without_smote', False)]:
        steps = [('power', PowerTransformer()), ('scale', StandardScaler())]
        if use_smote:
            steps.append(('smote', SMOTE(random_state=42)))
        steps.append(('tree', DecisionTreeClassifier(random_state=42)))
        search = GridSearchCV(Pipeline(steps), grid, scoring='recall', cv=cv, n_jobs=1, refit=True)
        search.fit(xtr, ytr)
        pred = search.predict(xte)
        results[key] = {
            'train_rows': int(len(xtr)), 'test_rows': int(len(xte)),
            'best_params': search.best_params_, 'cv_malignant_recall': float(search.best_score_),
            'accuracy': float(accuracy_score(yte, pred)),
            'balanced_accuracy': float(balanced_accuracy_score(yte, pred)),
            'malignant_recall': float(recall_score(yte, pred)),
            'malignant_f1': float(f1_score(yte, pred)),
            'confusion_matrix': confusion_matrix(yte, pred, labels=[0, 1]).tolist(),
            'classification_report': classification_report(yte, pred, labels=[0, 1], target_names=['Benign', 'Malignant'], output_dict=True, zero_division=0)
        }
        print(f"{key}: accuracy={results[key]['accuracy']:.4f}, malignant recall={results[key]['malignant_recall']:.4f}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'model_comparison.json').write_text(json.dumps(results, indent=2))
    rows = [{'configuration': key, **{k: val for k, val in rec.items() if k in ['accuracy', 'balanced_accuracy', 'malignant_recall', 'malignant_f1', 'cv_malignant_recall']}} for key, rec in results.items()]
    pd.DataFrame(rows).to_csv(out / 'model_comparison.csv', index=False)
    return results

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=root / 'data' / 'data.csv')
    parser.add_argument('--out', type=Path, default=root / 'results')
    args = parser.parse_args()
    evaluate(args.data, args.out)
