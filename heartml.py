"""
heartml.py - shared code for the SIT307 HD Machine Learning Mini Research task.

Part 1: reproduction of Bhagat, Sharma & Agarwal (2025), "An efficient stacking-based
        ensemble technique for early heart attack prediction", Multimedia Tools and
        Applications 84:36351-36375.
Part 2: LACS - Leakage-Aware, Clinically-encoded, Convex Stacking.

Author: Akshit Bhullar
"""
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

TARGET = "target"
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
NOMINAL = ["cp", "restecg", "slope", "thal"]
BINARY = ["sex", "fbs", "exang"]
METRICS = ["Accuracy", "Precision", "Recall", "F1", "AUC"]


# --------------------------------------------------------------------------- data
def load_data(path="data/heart.csv", deduplicate=False):
    df = pd.read_csv(path)
    if deduplicate:
        df = df.drop_duplicates().reset_index(drop=True)
    return df.drop(columns=TARGET), df[TARGET].values


def duplicate_report(path="data/heart.csv"):
    df = pd.read_csv(path)
    return {"rows": len(df), "duplicate_rows": int(df.duplicated().sum()),
            "unique_rows": int(len(df.drop_duplicates())),
            "unique_rows_with_conflicting_labels": int(
                df.drop_duplicates().drop(columns=TARGET).duplicated().sum())}


def test_rows_seen_in_train(Xtr, Xte):
    """Boolean mask: which test rows have an exact feature-level copy in the training set."""
    keys = set(map(tuple, Xtr.values))
    return np.array([tuple(r) in keys for r in Xte.values])


def scores(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred),
            "AUC": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else np.nan,
            "LogLoss": log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
            "Brier": brier_score_loss(y, proba)}


# --------------------------------------------------------------- Part 1: the paper
def paper_base_models(seed=0):
    """The six classifiers named in the paper. Hyperparameters are not reported, so
    library defaults are used (documented assumption). Scaling = 'standardising the numbers'."""
    return [
        ("LR", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))),
        ("DT", DecisionTreeClassifier(random_state=seed)),
        ("RF", RandomForestClassifier(random_state=seed)),
        ("XGB", XGBClassifier(random_state=seed, eval_metric="logloss")),
        ("NB", GaussianNB()),
        ("KNN", make_pipeline(StandardScaler(), KNeighborsClassifier())),
    ]


def paper_stacking(seed=0):
    """Stacking of the six base models with a logistic-regression meta-classifier
    trained on 5-fold out-of-fold base predictions (assumed; meta-learner not specified)."""
    return StackingClassifier(paper_base_models(seed), final_estimator=LogisticRegression(max_iter=1000),
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed), stack_method="predict_proba")


# ------------------------------------------------------------ Part 2: LACS method
class ClinicalMissingCodes(BaseEstimator, TransformerMixin):
    """In this release of the UCI data, the original missing values of `ca` and `thal`
    were silently recoded as ca=4 and thal=0 (codes that do not exist in the UCI
    codebook). Convert them back to NaN so they are imputed inside each training fold."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.loc[X["ca"] == 4, "ca"] = np.nan
        X.loc[X["thal"] == 0, "thal"] = np.nan
        return X


def clinical_preprocessor():
    """Clinically-typed encoding: nominal codes (chest-pain type, ECG, slope, thal) are
    one-hot encoded instead of being treated as ordered numbers; numeric features are
    median-imputed and standardised; binary flags pass through. All statistics are
    learned on the training fold only."""
    ct = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUMERIC),
        ("nom", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore", sparse_output=False)), NOMINAL),
        ("bin", "passthrough", BINARY),
    ])
    return make_pipeline(ClinicalMissingCodes(), ct)


def lacs_base_models(seed=0):
    """Diverse, deliberately regularised base learners for small-n tabular data."""
    P = clinical_preprocessor
    return [
        ("LR", make_pipeline(P(), LogisticRegressionCV(Cs=np.logspace(-3, 2, 12), cv=5, scoring="neg_log_loss",
                                                       max_iter=5000))),
        ("SVM", make_pipeline(P(), SVC(C=0.5, probability=True, random_state=seed))),
        ("KNN", make_pipeline(P(), KNeighborsClassifier(n_neighbors=15, weights="distance"))),
        ("NB", make_pipeline(P(), GaussianNB())),
        ("RF", make_pipeline(P(), RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                         random_state=seed, n_jobs=-1))),
        ("ET", make_pipeline(P(), ExtraTreesClassifier(n_estimators=500, min_samples_leaf=3,
                                                       random_state=seed, n_jobs=-1))),
        ("XGB", make_pipeline(P(), XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8,
                                                 colsample_bytree=0.8, reg_lambda=2.0, random_state=seed,
                                                 eval_metric="logloss", n_jobs=1))),
    ]


class ConvexStacking(BaseEstimator, ClassifierMixin):
    """Stacking with a *constrained* meta-learner.

    Instead of an unconstrained logistic-regression meta-classifier (which can put large
    positive/negative weights on highly correlated base predictions and overfit when n is
    small), the meta-learner here is a convex combination of base-model probabilities:
        p(x) = sum_k w_k p_k(x),   w_k >= 0,  sum_k w_k = 1.
    Weights minimise out-of-fold log-loss (softmax parameterisation) and are then shrunk
    toward the uniform average by factor `shrink` (a James-Stein-style guard against
    weight-estimation noise).  shrink=1 -> soft voting; shrink=0 -> pure optimised weights.
    """
    def __init__(self, base_fn=lacs_base_models, inner_folds=5, shrink=0.5, seed=0):
        self.base_fn, self.inner_folds, self.shrink, self.seed = base_fn, inner_folds, shrink, seed

    def fit(self, X, y):
        pool = self.base_fn(self.seed)
        cv = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed)
        P = np.column_stack([cross_val_predict(clone(m), X, y, cv=cv, method="predict_proba")[:, 1]
                             for _, m in pool])
        k = P.shape[1]

        def objective(v):
            w = np.exp(v - v.max()); w /= w.sum()
            return log_loss(y, np.clip(P @ w, 1e-6, 1 - 1e-6))

        v = minimize(objective, np.zeros(k), method="L-BFGS-B").x
        w = np.exp(v - v.max()); w /= w.sum()
        self.raw_weights_ = w
        self.weights_ = (1 - self.shrink) * w + self.shrink / k
        self.names_ = [n for n, _ in pool]
        self.oof_ = P
        self.bases_ = [clone(m).fit(X, y) for _, m in pool]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.column_stack([b.predict_proba(X)[:, 1] for b in self.bases_]) @ self.weights_
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# --------------------------------------------------------------- statistics
def corrected_resampled_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio (2003) corrected resampled t-test for repeated k-fold CV.
    a, b: per-fold scores of two models on identical folds. Returns (mean diff, t, p)."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return d.mean(), np.nan, np.nan
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=J - 1)
    return d.mean(), t, p
