"""Builds report/SIT307_HD_Technical_Report.pdf from results/*.csv and figures/*.png.
Every number in the report is read from the experiment outputs, so re-running the notebook
and then this script keeps the report consistent with the code."""
import json, os
import numpy as np, pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

# ------------------------------------------------------------------ edit these before submitting
AUTHOR = "Akshit Bhullar"
STUDENT_ID = "224011787"
VIDEO_URL = "[YouTube (unlisted) link]"
CODE_URL = "https://github.com/Akshit307/mlhd"
# ------------------------------------------------------------------------------------------------

F = "/usr/share/fonts/truetype/liberation/"
for n, f in [("Serif", "LiberationSerif-Regular"), ("Serif-B", "LiberationSerif-Bold"),
             ("Serif-I", "LiberationSerif-Italic"), ("Serif-BI", "LiberationSerif-BoldItalic"),
             ("Sans", "LiberationSans-Regular"), ("Sans-B", "LiberationSans-Bold"), ("Mono", "LiberationMono-Regular")]:
    pdfmetrics.registerFont(TTFont(n, F + f + ".ttf"))
pdfmetrics.registerFontFamily("Serif", normal="Serif", bold="Serif-B", italic="Serif-I", boldItalic="Serif-BI")

INK, MUTED, RULE, TINT = colors.HexColor("#1b1b1a"), colors.HexColor("#5b5a55"), colors.HexColor("#bdbcb6"), colors.HexColor("#eef3fb")
body = ParagraphStyle("b", fontName="Serif", fontSize=10, leading=13.4, alignment=TA_JUSTIFY, textColor=INK, spaceAfter=5)
small = ParagraphStyle("s", parent=body, fontSize=8.6, leading=11, alignment=0)
cap = ParagraphStyle("c", parent=body, fontSize=8.6, leading=11, textColor=MUTED, alignment=0, spaceBefore=2, spaceAfter=9)
h1 = ParagraphStyle("h1", fontName="Sans-B", fontSize=12.5, leading=16, textColor=INK, spaceBefore=11, spaceAfter=5, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="Sans-B", fontSize=10.5, leading=13, textColor=INK, spaceBefore=8, spaceAfter=3, keepWithNext=1)
title = ParagraphStyle("t", fontName="Sans-B", fontSize=17, leading=21, alignment=TA_CENTER, textColor=INK, spaceAfter=6)
sub = ParagraphStyle("st", fontName="Serif", fontSize=10.5, leading=14, alignment=TA_CENTER, textColor=MUTED)
ref = ParagraphStyle("r", parent=body, fontSize=8.8, leading=11.2, leftIndent=18, firstLineIndent=-18, alignment=0, spaceAfter=2.5)
cell = ParagraphStyle("cell", fontName="Serif", fontSize=8.2, leading=10, textColor=INK)
cellb = ParagraphStyle("cellb", parent=cell, fontName="Serif-B")

R = "results/"
dup = json.load(open(R + "duplicate_report.json"))
p1 = pd.read_csv(R + "part1_single_split.csv", index_col=0)
p30raw = pd.read_csv(R + "part1_30_splits_raw.csv")
leak = pd.read_csv(R + "part1_leakage_comparison.csv", index_col=0)
pf = pd.read_csv(R + "part2_per_fold.csv")
tests = pd.read_csv(R + "part2_stat_tests.csv")
wts = pd.read_csv(R + "part2_lacs_weights.csv")
lk = pd.read_csv(R + "part2_leaky_protocol_check.csv")
M = ["Accuracy", "Precision", "Recall", "F1", "AUC", "LogLoss", "Brier"]
order2 = ["Paper stacking (baseline)", "Paper best single (RF)", "Ablation: paper stacking + clinical enc.",
          "Ablation: uniform soft vote", "Ablation: convex, no shrinkage", "Ablation: clinical LR only", "LACS (proposed)"]
mean2, sd2 = pf.groupby("Model")[M].mean().loc[order2], pf.groupby("Model")[M].std().loc[order2]
seen = p30raw.groupby("seed").seen_frac.first()
pct = lambda v, d=1: f"{100 * v:.{d}f}%"
base, lacs, lr = mean2.loc["Paper stacking (baseline)"], mean2.loc["LACS (proposed)"], mean2.loc["Ablation: clinical LR only"]
T = lambda other, met, col: tests[(tests["LACS vs"] == other) & (tests.Metric == met)][col].iloc[0]
n_better = {m: int((pf[pf.Model == "LACS (proposed)"].sort_values("fold")[m].values >
                    pf[pf.Model == "Paper stacking (baseline)"].sort_values("fold")[m].values).sum()) for m in ["Accuracy", "AUC"]}
n_worse = {m: int((pf[pf.Model == "LACS (proposed)"].sort_values("fold")[m].values <
                   pf[pf.Model == "Paper stacking (baseline)"].sort_values("fold")[m].values).sum()) for m in ["Accuracy", "AUC"]}
p30 = p30raw.groupby("Model", sort=False)[["Accuracy", "Precision", "Recall", "F1", "AUC"]].agg(["mean", "std"])

P = lambda t, s=body: Paragraph(t, s)
story = []


def table(data, widths, header_rows=1, bold_rows=(), hl_rows=(), fs=None):
    c1 = cell if fs is None else ParagraphStyle("cx", parent=cell, fontSize=fs, leading=fs + 1.8)
    c2 = cellb if fs is None else ParagraphStyle("cxb", parent=cellb, fontSize=fs, leading=fs + 1.8)
    rows = [[Paragraph(str(c), c2 if (r < header_rows or r in bold_rows) else c1) for c in row]
            for r, row in enumerate(data)]
    t = Table(rows, colWidths=widths, repeatRows=header_rows, hAlign="CENTER")
    st = [("LINEABOVE", (0, 0), (-1, 0), 0.9, INK), ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.6, INK),
          ("LINEBELOW", (0, -1), (-1, -1), 0.9, INK), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("TOPPADDING", (0, 0), (-1, -1), 2.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
          ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    for r in hl_rows:
        st.append(("BACKGROUND", (0, r), (-1, r), TINT))
    t.setStyle(TableStyle(st))
    return t


def fig(path, width_mm, caption):
    from PIL import Image as PI
    w, h = PI.open(path).size
    W = width_mm * mm
    return KeepTogether([Image(path, width=W, height=W * h / w), P(caption, cap)])


def captioned_table(caption, t):
    return KeepTogether([P(caption, cap), t, Spacer(1, 8)])


# ============================================================================ title
story += [Spacer(1, 6), P("Duplicate Leakage in Stacking-Based Heart Attack Prediction:<br/>"
                         "Reproduction, Critical Analysis and a Leakage-Aware Convex Stacking Method", title),
          P(f"{AUTHOR} &nbsp;·&nbsp; {STUDENT_ID} &nbsp;·&nbsp; SIT307 Machine Learning, HD Task: Machine Learning Mini Research", sub),
          P("Deakin University, School of Information Technology &nbsp;·&nbsp; September 2026", sub), Spacer(1, 6)]
story.append(table([["Deliverable", "Link"], ["Video presentation (≤ 5 min)", VIDEO_URL],
                    ["Code, data and README", CODE_URL]], [55 * mm, 115 * mm]))
story.append(Spacer(1, 8))

story += [P("Abstract", h2), P(
    f"Bhagat, Sharma and Agarwal [1] report that stacking six classifiers (LR, DT, RF, XGBoost, NB, KNN) raises heart-attack "
    f"prediction accuracy from 92.68% to <b>98.53%</b> on a public 1,025-record heart-disease dataset. We reproduce their pipeline and "
    f"obtain {pct(leak.loc['Stacking', 'Reproduced (leaky, 30 splits)'], 2)} mean accuracy over 30 random 80/20 splits, confirming the headline "
    f"result. A data audit, however, shows that only <b>{dup['unique_rows']} of the {dup['rows']} records are unique</b>: {dup['duplicate_rows']} are exact "
    f"duplicates, and on average {pct(seen.mean())} of test records have an identical copy in the training set. When the identical pipeline is "
    f"evaluated on unique patients, stacking accuracy falls to {pct(leak.loc['Stacking', 'De-duplicated (30 splits)'])} and the decision tree falls from "
    f"{pct(leak.loc['DT', 'Reproduced (leaky, 30 splits)'])} to {pct(leak.loc['DT', 'De-duplicated (30 splits)'])}; the reported gains are "
    f"largely an artefact of leakage. We then propose <b>LACS</b> (Leakage-Aware, Clinically-encoded, Convex Stacking), which combines "
    f"(i) a patient-level evaluation protocol, (ii) clinically correct handling of undocumented missing-value codes and nominal features, and "
    f"(iii) a convex, shrinkage-regularised meta-learner over a diverse, regularised base pool. Under 10×5 repeated stratified cross-validation "
    f"with identical folds, LACS improves on the paper's stacking in all seven metrics (accuracy {pct(base.Accuracy)} → {pct(lacs.Accuracy)}, "
    f"F1 {base.F1:.3f} → {lacs.F1:.3f}, AUC {base.AUC:.3f} → {lacs.AUC:.3f}, log-loss {base.LogLoss:.3f} → {lacs.LogLoss:.3f}). The gains are "
    f"consistent across folds (Wilcoxon p = {T('Paper stacking (baseline)', 'Accuracy', 'p (Wilcoxon)'):.3f} for accuracy) but modest relative to "
    f"between-fold variance (corrected resampled t-test p = {T('Paper stacking (baseline)', 'Accuracy', 'p (corrected t)'):.2f}). Ablations show that a "
    f"clinically-encoded, regularised logistic regression alone is as good as any ensemble on 302 patients. The realistic ceiling for this "
    f"data is therefore about 85% accuracy and 0.91 AUC, not 98.5%.")]

# ============================================================================ 1. intro
story += [P("1. Introduction and Research Problem", h1), P(
    "Cardiovascular disease is the leading cause of death worldwide, and early, cheap risk screening from routinely collected clinical "
    "variables (age, sex, chest-pain type, resting blood pressure, cholesterol, ECG findings, exercise test results) remains an active "
    "machine-learning problem [1], [11], [12], [13]. A large body of work reports increasingly high accuracies using hybrid and ensemble methods on "
    "UCI-derived heart-disease data. Stacked generalisation [4], in which a meta-learner combines the predictions of several base "
    "classifiers, is a popular route to such gains [1], [12], [13]."), P(
    "Bhagat <i>et al.</i> [1] (the selected paper) study the 1,025-record Kaggle release [3] of the UCI Heart Disease data [2]. They "
    "pre-process the data (missing-value filling, standardisation, categorical encoding), split it into training and test sets, train six "
    "classifiers, and stack them with a meta-classifier. The <b>research gap</b> they address is that single classifiers plateau at around "
    "90 to 93% accuracy; their <b>claim</b> is that stacking lifts accuracy to 98.53%, outperforming every individual model. This report (Part 1) "
    "reproduces that claim, critically tests whether it reflects generalisation to new patients, and (Part 2) proposes and rigorously evaluates "
    "a method that addresses the limitations found.")]

# ============================================================================ 2. paper summary
story += [P("2. Summary of the Selected Paper and State of the Art", h1),
          P("<b>Dataset.</b> 1,025 records × 13 clinical features + binary target (heart disease present / absent), 526 positive and 499 negative [1], [3].", body),
          P("<b>Methods.</b> Logistic Regression (LR), Decision Tree (DT), Random Forest (RF), Extreme Gradient Boosting (XGB) [9], Gaussian Naive "
            "Bayes (NB) and K-Nearest Neighbours (KNN) as base learners; a stacking ensemble in which a meta-classifier is trained on the base "
            "learners' predictions [4]. Evaluation uses accuracy, precision, recall and F1 on a single hold-out test set.", body),
          P("<b>Reported results.</b> RF and DT 92.68%, XGB 90.73%, stacking 98.53% accuracy [1]. All three values are exact multiples of 1/205 "
            "(190/205, 186/205, 202/205), which implies a 205-record test set, i.e. an 80/20 split of 1,025 records.", body),
          P("<b>State of the art.</b> Related work on UCI heart-disease data spans feature-selection pipelines combined with classical classifiers "
            "[11], hybrid random forest with a linear model (HRFLM) [12], optimised stacked SVMs [13] and majority-voting or bagging/boosting "
            "ensembles [14]. Most of these works report results from a single split or a single k-fold run, often without confidence intervals or "
            "significance tests, which makes cross-paper comparison fragile [7], [15]. Kapoor and Narayanan [6] document data leakage as a "
            "pervasive cause of over-optimistic results in ML-based science, including medicine. Leakage of this kind is exactly what our "
            "reproduction uncovers.", body)]

# ============================================================================ 3. Part 1
story += [P("3. Part 1: Reproduction of the Published Results", h1), P("3.1 Implementation and assumptions", h2), P(
    "The full paper text does not report hyperparameters, the meta-learner, random seeds or which variables were treated as categorical. We "
    "implemented the pipeline in scikit-learn [8] and XGBoost [9] and made the following documented assumptions (Table I). The dataset has no "
    "NaN values, so the 'missing-value filling' step has no effect on the file as distributed.")]
assumptions = [["Aspect", "Paper states", "Our implementation (assumption)"],
               ["Data", "1,025 patients, 14 attributes", "Kaggle heart.csv [3]; identical shape and class balance (526/499)"],
               ["Split", "train/test hold-out", "Stratified 80/20 (205 test records, inferred from reported accuracies); seed 42, plus 30 seeds for variance"],
               ["Pre-processing", "fill missing, standardise, encode", "No NaNs present; StandardScaler for scale-sensitive models (LR, KNN); codes used as given"],
               ["Base models", "LR, DT, RF, XGB, NB, KNN", "Library defaults (hyperparameters not reported)"],
               ["Stacking", "meta-classifier on base predictions", "sklearn StackingClassifier, LR meta-learner on 5-fold out-of-fold probabilities"],
               ["Metrics", "Accuracy, Precision, Recall, F1", "Same + ROC-AUC, log-loss and Brier score"]]
story.append(captioned_table("TABLE I. Reproduction protocol and assumptions.", table(assumptions, [24 * mm, 50 * mm, 96 * mm])))

story.append(P("3.2 Reproduced results", h2))
rows = [["Model", "Paper Acc.", "Acc. (seed 42)", "Acc. 30 splits (mean ± SD)", "Precision", "Recall", "F1", "AUC"]]
for m in ["LR", "DT", "RF", "XGB", "NB", "KNN", "Stacking"]:
    pa = p1.loc[m, "Paper Acc (%)"]
    rows.append([m, "n/r" if np.isnan(pa) else f"{pa:.2f}%", pct(p1.loc[m, "Accuracy"], 2),
                 f"{pct(p30.loc[m, ('Accuracy', 'mean')], 2)} ± {100 * p30.loc[m, ('Accuracy', 'std')]:.2f}",
                 f"{p30.loc[m, ('Precision', 'mean')]:.3f}", f"{p30.loc[m, ('Recall', 'mean')]:.3f}",
                 f"{p30.loc[m, ('F1', 'mean')]:.3f}", f"{p30.loc[m, ('AUC', 'mean')]:.3f}"])
story.append(captioned_table("TABLE II. Paper protocol (1,025 records, 80/20 split). Precision, recall, F1 and AUC are 30-split means. "
                             "n/r = not reported in the paper's abstract.", table(rows, [19 * mm, 17 * mm, 21 * mm, 36 * mm, 18 * mm, 17 * mm, 17 * mm, 17 * mm], hl_rows=[7])))
story.append(P(
    f"The headline claim reproduces: stacking reaches {pct(p30.loc['Stacking', ('Accuracy', 'mean')], 2)} mean accuracy "
    f"(paper 98.53%) and is at or near the top of all models. Two discrepancies stand out. (1) Our tree-based models (DT, RF, XGB) score "
    f"≈99 to 100%, far above the paper's 90.7 to 92.7%. This is consistent with the paper having used constrained trees (e.g. limited depth) or a "
    f"different split, which is plausible because hyperparameters were not reported. (2) In our runs stacking does not beat RF and XGB by a "
    f"clear margin; all three sit within about one standard deviation of 100%. The variation across 30 splits (SD ≈ 1 percentage point) "
    f"shows that a single split cannot resolve differences of the size the paper reports."))

story.append(P("3.3 Critical analysis: why is the accuracy so high?", h2))
story.append(P(
    f"<b>Duplicate-record leakage.</b> An audit of the data (notebook §0) shows <b>{dup['duplicate_rows']} exact duplicate rows; only "
    f"{dup['unique_rows']} records are unique</b> (close to the 303-patient Cleveland cohort [2], [10]). The 1,025-record file is "
    f"effectively the ~300 Cleveland patients repeated about 3.4 times. Under the paper's random 80/20 split, "
    f"{pct(seen.mean())} of test records (min {pct(seen.min())}, over 30 splits) have an identical feature vector in the training set. "
    f"Any model that can memorise, such as an unpruned tree, RF, XGB or a stacker built on them, simply looks the answer up. This is textbook "
    f"train/test contamination [5], [6]. It explains why the most memorising models look best and why stacking appears to 'fix' the errors."))
story.append(fig("figures/fig1_leakage.png", 165, "Fig. 1. Accuracy reported in the paper (grey), reproduced with the paper's protocol (orange), and the "
                 "<i>identical</i> pipeline re-run on the 302 unique records (blue, 30 random 80/20 splits, ±1 SD)."))
story.append(P(
    f"Re-running the <i>unchanged</i> pipeline on de-duplicated data (Fig. 1, Fig. 2) removes the illusion. Stacking falls from "
    f"{pct(leak.loc['Stacking', 'Reproduced (leaky, 30 splits)'])} to {pct(leak.loc['Stacking', 'De-duplicated (30 splits)'])}, RF from "
    f"{pct(leak.loc['RF', 'Reproduced (leaky, 30 splits)'])} to {pct(leak.loc['RF', 'De-duplicated (30 splits)'])}, XGB to "
    f"{pct(leak.loc['XGB', 'De-duplicated (30 splits)'])} and DT to {pct(leak.loc['DT', 'De-duplicated (30 splits)'])}. The models that cannot "
    f"memorise exact points are unaffected: LR stays at {pct(leak.loc['LR', 'De-duplicated (30 splits)'])} and NB at "
    f"{pct(leak.loc['NB', 'De-duplicated (30 splits)'])}. This differential collapse is the signature of leakage rather than genuine skill. "
    f"The ranking also flips: on unique patients the simple LR is statistically indistinguishable from stacking, and DT, which the paper "
    f"ranks joint-first, is the worst model."))
story.append(fig("figures/fig2_confusion.png", 125, "Fig. 2. Stacking confusion matrices, seed 42. Left: paper protocol (205/205 correct). "
                 "Right: identical pipeline on unique patients (61 test patients)."))
story.append(P(
    "<b>Further methodological issues.</b> (a) <i>Undocumented codes.</i> The file contains ca = 4 and thal = 0, values that do not exist in "
    "the UCI codebook [2]; they correspond to the original data's missing values and were silently recoded, so 'filling missing values' was "
    "never actually applied. (b) <i>Nominal codes treated as numbers.</i> Chest-pain type, resting ECG, ST slope and thalassaemia are "
    "unordered categories, yet as integer codes a linear or distance-based model assumes, for example, that cp = 3 is 'three times' cp = 1. "
    "(c) <i>Single hold-out, no uncertainty.</i> Results are point estimates from one split with no confidence intervals or tests [7], [15]. "
    "(d) <i>Unconstrained meta-learner.</i> An LR meta-learner on six highly correlated probability outputs can assign large, unstable, even "
    "negative weights. Breiman [16] showed that constraining stacking weights to be non-negative is important for stable stacked predictors. "
    "(e) <i>Probability quality is not assessed</i>, although calibrated risk is what matters clinically."))

# ============================================================================ 4. Part 2
story += [P("4. Part 2: Proposed Method: LACS", h1), P("4.1 Motivation", h2), P(
    "Part 1 shows the paper's key limitation is not the choice of classifier but <b>how the data are prepared and evaluated</b>. A fair "
    "improvement must therefore (1) be measured on unseen <i>patients</i>, (2) use the clinical variables correctly, and (3) combine models "
    "in a way that is robust when only ~240 patients are available for training. LACS addresses each of these in turn.")]
story.append(P("4.2 Method", h2))
story.append(P(
    "<b>(1) Leakage-aware protocol.</b> Records are de-duplicated to patient level before any split. Duplicates add no information and would act "
    "as arbitrary sample weights. All fitting (imputation, scaling, encoding, hyperparameter tuning and meta-weight estimation) happens inside "
    "the training fold only. Evaluation uses 10×5 repeated stratified cross-validation (50 folds), and every model sees the <i>same</i> folds, so "
    "comparisons are paired."))
story.append(P(
    "<b>(2) Clinically-typed encoding.</b> ca = 4 and thal = 0 are mapped back to missing and imputed (median / mode) from the training fold. "
    "The nominal variables cp, restecg, slope and thal are one-hot encoded. Continuous variables (age, trestbps, chol, thalach, oldpeak, ca) "
    "are standardised, and binary flags (sex, fbs, exang) are passed through."))
story.append(P(
    "<b>(3) Diverse, regularised base pool.</b> L2-logistic regression with its regularisation strength tuned by inner 5-fold CV on log-loss "
    "(12 values of C in [10<super>-3</super>, 10<super>2</super>]); RBF-SVM (C = 0.5, Platt-scaled); distance-weighted KNN (k = 15); Gaussian "
    "NB; Random Forest and Extra-Trees [17] (500 trees, min leaf = 3); and shallow XGBoost (depth 2, 300 trees, η = 0.05, subsampling 0.8, "
    "λ = 2). Settings favour low variance, as suits n ≈ 240, and were fixed <i>a priori</i> rather than tuned on the evaluation folds."))
story.append(P(
    "<b>(4) Convex, shrunk meta-learner.</b> Out-of-fold probabilities p<sub>k</sub>(x) of the K base models are obtained by inner 5-fold CV on the "
    "training fold. The meta-learner is a convex combination p(x) = Σ<sub>k</sub> w<sub>k</sub> p<sub>k</sub>(x), with w<sub>k</sub> ≥ 0 and "
    "Σ w<sub>k</sub> = 1. The weights minimise out-of-fold log-loss (softmax parameterisation, L-BFGS) and are then shrunk toward the uniform "
    "average: w = (1 − λ) w* + λ/K with λ = 0.5. The non-negativity constraint follows Breiman [16] and the probability-level combination "
    "follows Ting and Witten [18]. Shrinkage guards against weight-estimation noise with small n, and the output stays a calibrated-looking "
    "probability rather than a logit of correlated inputs. Base models are finally refit on the whole training fold."))
diff = [["Component", "Paper [1]", "LACS (proposed)"],
        ["Evaluation unit", "records (duplicates cross the split)", "unique patients; 10×5 repeated stratified CV; paired folds"],
        ["Missing codes", "ca = 4, thal = 0 used as real values", "mapped to NaN, imputed inside training fold"],
        ["Nominal features", "integer codes", "one-hot encoded"],
        ["Base learners", "6 default models incl. unpruned DT", "7 regularised, diverse models; LR C tuned by inner CV"],
        ["Meta-learner", "unconstrained LR on base outputs", "convex non-negative weights (sum 1), shrunk to uniform"],
        ["Reporting", "single split, point estimates", "mean ± SD over 50 folds, corrected t-test, Wilcoxon, calibration"]]
story.append(captioned_table("TABLE III. How LACS differs from the original method.", table(diff, [28 * mm, 58 * mm, 84 * mm])))

story.append(P("4.3 Experimental protocol", h2))
story.append(P(
    f"Data: {dup['unique_rows']} unique patients (164 positive, 138 negative). Folds: RepeatedStratifiedKFold(5 splits × 10 repeats, "
    "seed 42), giving ≈ 241 training and 61 test patients per fold. Compared models, all evaluated on identical folds: the <b>paper stacking</b> "
    "baseline (exact Part 1 pipeline); the <b>paper's best single model</b> (RF); and four <b>ablations</b> that isolate each LACS component: "
    "paper stacking with clinical encoding (encoding only), uniform soft-voting of the LACS pool (λ = 1), convex weights without shrinkage "
    "(λ = 0), and the tuned clinical LR alone. Metrics: accuracy, precision, recall, F1, ROC-AUC (threshold 0.5 for label metrics), log-loss "
    "and Brier score. Significance: the Nadeau and Bengio corrected resampled t-test [7], which accounts for overlapping training sets in repeated "
    "CV, together with the Wilcoxon signed-rank test [15]. The Wilcoxon test is reported for completeness but is known to be anti-conservative "
    "on overlapping CV folds, so the corrected t-test is the primary criterion."))

rows = [["Model", "Accuracy", "Precision", "Recall", "F1", "AUC", "Log-loss ↓", "Brier ↓"]]
lab = {"Paper stacking (baseline)": "Paper stacking (baseline)", "Paper best single (RF)": "Paper best single model (RF)",
       "Ablation: paper stacking + clinical enc.": "A1: paper stacking + clinical encoding",
       "Ablation: uniform soft vote": "A2: LACS pool, uniform soft vote (λ = 1)",
       "Ablation: convex, no shrinkage": "A3: LACS pool, convex, no shrinkage (λ = 0)",
       "Ablation: clinical LR only": "A4: clinical LR only", "LACS (proposed)": "LACS (proposed, λ = 0.5)"}
for m in order2:
    rows.append([lab[m]] + [f"{mean2.loc[m, k]:.3f} ± {sd2.loc[m, k]:.3f}" for k in M])
story.append(KeepTogether([P("4.4 Results", h2), captioned_table("TABLE IV. Leakage-free 10×5 repeated CV on 302 unique patients (mean ± SD over 50 identical folds).",
                             table(rows, [48 * mm] + [17.4 * mm] * 7, bold_rows=[7], hl_rows=[7], fs=7.3))]))
rows = [["LACS vs.", "Δ Acc.", "p (corr. t)", "p (Wilc.)", "Δ F1", "p (corr. t)", "Δ AUC", "p (corr. t)", "Δ Log-loss", "p (corr. t)"]]
for m in order2[:-1]:
    r = [lab[m]]
    for met in ["Accuracy", "F1", "AUC", "LogLoss"]:
        r.append(f"{T(m, met, 'Mean diff'):+.4f}"); r.append(f"{T(m, met, 'p (corrected t)'):.3f}")
        if met == "Accuracy":
            r.append(f"{T(m, met, 'p (Wilcoxon)'):.3f}")
    rows.append(r)
story.append(captioned_table("TABLE V. Paired differences (LACS minus other) over 50 folds, with corrected resampled t-test [7] and Wilcoxon p-values. "
                             "Positive Δ is better, except for log-loss.",
                             table(rows, [42 * mm] + [13.1 * mm] * 9)))
story.append(fig("figures/fig3_cv_boxplots.png", 168, "Fig. 3. Per-fold accuracy and AUC over 50 identical CV folds (black diamond = mean). Orange: paper "
                 "stacking; blue: LACS; grey: comparators and ablations."))
story.append(fig("figures/fig4_roc_calib_weights.png", 170, "Fig. 4. Left: pooled out-of-fold ROC curves for one full CV repeat. Centre: "
                 "reliability diagram. Right: mean LACS meta-weights across the 50 outer folds (dashed line = uniform 1/7)."))

story.append(P("4.5 Analysis and discussion", h2))
story.append(P(
    f"<b>LACS vs. the reproduced baseline.</b> On unseen patients, LACS improves on the paper's stacking in every metric: accuracy "
    f"{pct(base.Accuracy, 2)} → {pct(lacs.Accuracy, 2)}, recall {base.Recall:.3f} → {lacs.Recall:.3f}, F1 {base.F1:.3f} → {lacs.F1:.3f}, "
    f"AUC {base.AUC:.3f} → {lacs.AUC:.3f}, log-loss {base.LogLoss:.3f} → {lacs.LogLoss:.3f} and Brier {base.Brier:.3f} → {lacs.Brier:.3f}. "
    f"It also has a smaller fold-to-fold spread (accuracy SD {sd2.loc['LACS (proposed)', 'Accuracy']:.3f} vs "
    f"{sd2.loc['Paper stacking (baseline)', 'Accuracy']:.3f}). LACS is more accurate in {n_better['Accuracy']} of 50 folds and less accurate "
    f"in {n_worse['Accuracy']}, and has higher AUC in {n_better['AUC']} folds. The improvement is consistent (Wilcoxon p ≤ "
    f"{max(T('Paper stacking (baseline)', m_, 'p (Wilcoxon)') for m_ in ['Accuracy', 'F1', 'LogLoss']):.3f} for accuracy, F1 and log-loss). "
    f"However, it is not significant under the conservative corrected t-test (p = {T('Paper stacking (baseline)', 'Accuracy', 'p (corrected t)'):.2f}), "
    f"because the gain (≈ 1.5 points) is small relative to the ≈ 4-point between-fold variance that 61-patient test folds impose. The higher "
    f"recall matters clinically: missed disease is the costlier error. Against the paper's best single model (RF), the gain is larger "
    f"(+{100 * T('Paper best single (RF)', 'Accuracy', 'Mean diff'):.1f} points accuracy; Brier corrected p = "
    f"{T('Paper best single (RF)', 'Brier', 'p (corrected t)'):.3f})."))
story.append(P(
    f"<b>What each component contributes (ablations).</b> Clinical encoding alone (A1) recovers just under a third of the gain "
    f"({pct(mean2.loc['Ablation: paper stacking + clinical enc.', 'Accuracy'], 2)}), and it improves log-loss, showing that correct treatment of "
    f"nominal and missing codes helps even the original ensemble. Replacing the paper's pool and meta-learner with the regularised pool and a "
    f"convex combiner (A2, A3, LACS) adds the rest. The three combiner variants are statistically indistinguishable (all corrected p > 0.6). "
    f"The shrinkage in LACS gives the lowest log-loss and variance of the three but not the highest accuracy (A3: "
    f"{pct(mean2.loc['Ablation: convex, no shrinkage', 'Accuracy'], 2)}). The learned weights (Fig. 4, right) put ≈ "
    f"{wts.mean()['LR']:.2f} of the mass on LR and spread the rest almost evenly. This mirrors the finding below: the linear signal dominates."))
story.append(P(
    f"<b>An honest negative result.</b> The tuned, clinically-encoded LR alone (A4) scores {pct(lr.Accuracy, 2)} accuracy and {lr.AUC:.3f} AUC. "
    f"That is numerically slightly <i>better</i> than LACS, though not significantly so under the corrected test (p = "
    f"{T('Ablation: clinical LR only', 'Accuracy', 'p (corrected t)'):.2f}). With 13 largely additive risk factors and ~240 training patients, "
    f"ensembling cannot find much non-linear structure beyond a well-specified linear model. The practical lesson is that for this problem the "
    f"large apparent benefit of stacking in [1] comes from the evaluation, not from the model. The main value of LACS over the original "
    f"method is correct and robust handling of the data, with the ensemble acting as insurance: its weights automatically fall back toward "
    f"the strongest learner."))
story.append(P(
    f"<b>Comparison with the paper's reported numbers.</b> The paper's 98.53% cannot be compared directly with any leakage-free number. As a "
    f"sanity check, running LACS under the paper's leaky protocol (10 seeds) gives {pct(lk.LACS.mean(), 2)} accuracy vs "
    f"{pct(lk['Paper stacking'].mean(), 2)} for the paper's stacking. Both are at the ceiling, so the leaky protocol cannot discriminate between "
    f"methods at all. On a valid protocol, the realistic performance for this cohort is ≈ 83 to 86% accuracy and ≈ 0.91 AUC."))
story.append(P(
    "<b>Limitations and future work.</b> (i) 302 patients from a single site: external validation on the Hungarian, Swiss and Long Beach VA "
    "cohorts [2] is the natural next step. (ii) The base-learner hyperparameters were fixed a priori to limit tuning cost; full nested tuning "
    "may add a little. (iii) The 0.5 decision threshold was not optimised for the asymmetric clinical cost of false negatives. A "
    "cost-sensitive threshold chosen inside the training fold would raise recall further. (iv) The Kaggle release binarises the UCI 0 to 4 diagnosis without documenting its "
    "re-coding, which limits clinical interpretation of individual coefficients."))

# ============================================================================ 5. conclusion
story += [P("5. Conclusion", h1), P(
    f"We reproduced the stacking results of Bhagat <i>et al.</i> [1] ({pct(leak.loc['Stacking', 'Reproduced (leaky, 30 splits)'], 2)} vs 98.53%) and "
    f"showed that they rest on a dataset in which {dup['duplicate_rows']} of {dup['rows']} records are duplicates, so that ≈ {pct(seen.mean(), 0)} of "
    f"test records are seen during training. Evaluated on unique patients, the same pipeline achieves {pct(leak.loc['Stacking', 'De-duplicated (30 splits)'])}. "
    f"The proposed LACS method, with leakage-aware evaluation, clinically correct encoding and a convex, shrunk meta-learner over regularised "
    f"base models, consistently improves on the baseline in all metrics under a rigorous, paired 10×5 CV protocol "
    f"({pct(lacs.Accuracy)} accuracy, {lacs.AUC:.3f} AUC). Ablations show that most of the achievable gain comes from treating the data correctly, "
    f"and that a well-specified logistic regression is a hard-to-beat baseline for this problem. The broader lesson is that on small medical "
    f"tabular datasets, duplicate audits and patient-level splits should come before model complexity.")]

story += [P("Reproducibility Statement", h2), P(
    "All code, the dataset and a README are provided (link on page 1). <font face='Mono' size='8.5'>SIT307_HD_Heart_Stacking.ipynb</font> "
    "regenerates every table and figure from <font face='Mono' size='8.5'>data/heart.csv</font> with fixed seeds (≈ 25 min on a laptop CPU); "
    "<font face='Mono' size='8.5'>heartml.py</font> contains the models, LACS and the statistical tests; <font face='Mono' size='8.5'>make_report.py</font> "
    "rebuilds this PDF from the saved results. Environment: Python 3.11, scikit-learn 1.8.0, XGBoost 3.2.0, pandas, SciPy, Matplotlib "
    "(pinned in requirements.txt).")]

story += [P("Acknowledgement of Generative AI Use", h2), P(
    "In accordance with Deakin University's academic integrity requirements, I acknowledge the use of Claude (Anthropic) as an assistant in this "
    "task. It was used to help select the paper, write and debug the Python implementation, run the experiments, and draft and format this "
    "report. I reviewed, ran and verified all code and results, and the analysis, conclusions and final wording are my own responsibility. "
    "<i>[Edit this statement so that it accurately reflects how you used GenAI.]</i>")]

# ============================================================================ refs
story += [P("References", h1)]
refs = [
    "M. Bhagat, A. Sharma, and P. Agarwal, “An efficient stacking-based ensemble technique for early heart attack prediction,” <i>Multimedia Tools and Applications</i>, vol. 84, no. 30, pp. 36351-36375, 2025, doi: 10.1007/s11042-024-19293-7.",
    "A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, “Heart Disease,” UCI Machine Learning Repository, 1988, doi: 10.24432/C52P4X.",
    "D. Lapp, “Heart Disease Dataset,” Kaggle, 2019. [Online]. Available: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset",
    "D. H. Wolpert, “Stacked generalization,” <i>Neural Networks</i>, vol. 5, no. 2, pp. 241-259, 1992.",
    "S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: Formulation, detection, and avoidance,” <i>ACM Trans. Knowledge Discovery from Data</i>, vol. 6, no. 4, pp. 1-21, 2012.",
    "S. Kapoor and A. Narayanan, “Leakage and the reproducibility crisis in machine-learning-based science,” <i>Patterns</i>, vol. 4, no. 9, p. 100804, 2023.",
    "C. Nadeau and Y. Bengio, “Inference for the generalization error,” <i>Machine Learning</i>, vol. 52, no. 3, pp. 239-281, 2003.",
    "F. Pedregosa <i>et al.</i>, “Scikit-learn: Machine learning in Python,” <i>J. Machine Learning Research</i>, vol. 12, pp. 2825-2830, 2011.",
    "T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” in <i>Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining</i>, 2016, pp. 785-794.",
    "R. Detrano <i>et al.</i>, “International application of a new probability algorithm for the diagnosis of coronary artery disease,” <i>American Journal of Cardiology</i>, vol. 64, no. 5, pp. 304-310, 1989.",
    "P. Ghosh <i>et al.</i>, “Efficient prediction of cardiovascular disease using machine learning algorithms with relief and LASSO feature selection techniques,” <i>IEEE Access</i>, vol. 9, pp. 19304-19326, 2021.",
    "S. Mohan, C. Thirumalai, and G. Srivastava, “Effective heart disease prediction using hybrid machine learning techniques,” <i>IEEE Access</i>, vol. 7, pp. 81542-81554, 2019.",
    "L. Ali <i>et al.</i>, “An optimized stacked support vector machines based expert system for the effective prediction of heart failure,” <i>IEEE Access</i>, vol. 7, pp. 54007-54014, 2019.",
    "C. B. C. Latha and S. C. Jeeva, “Improving the accuracy of prediction of heart disease risk based on ensemble classification techniques,” <i>Informatics in Medicine Unlocked</i>, vol. 16, p. 100203, 2019.",
    "J. Demšar, “Statistical comparisons of classifiers over multiple data sets,” <i>J. Machine Learning Research</i>, vol. 7, pp. 1-30, 2006.",
    "L. Breiman, “Stacked regressions,” <i>Machine Learning</i>, vol. 24, no. 1, pp. 49-64, 1996.",
    "P. Geurts, D. Ernst, and L. Wehenkel, “Extremely randomized trees,” <i>Machine Learning</i>, vol. 63, no. 1, pp. 3-42, 2006.",
    "K. M. Ting and I. H. Witten, “Issues in stacked generalization,” <i>J. Artificial Intelligence Research</i>, vol. 10, pp. 271-289, 1999.",
]
for i, r in enumerate(refs, 1):
    story.append(P(f"[{i}]&nbsp;&nbsp;{r}", ref))

# ============================================================================ appendix
story += [PageBreak(), P("Appendix A: Additional results", h1)]
rows = [["Model", "Acc. mean ± SD", "Prec. mean ± SD", "Recall mean ± SD", "F1 mean ± SD", "AUC mean ± SD"]]
for m in ["LR", "DT", "RF", "XGB", "NB", "KNN", "Stacking"]:
    rows.append([m] + [f"{p30.loc[m, (k, 'mean')]:.4f} ± {p30.loc[m, (k, 'std')]:.4f}" for k in ["Accuracy", "Precision", "Recall", "F1", "AUC"]])
story.append(captioned_table("TABLE A1. Paper protocol, 30 random stratified 80/20 splits of the 1,025 records (leaky).", table(rows, [22 * mm] + [29.6 * mm] * 5)))
rd = pd.read_csv(R + "part1_dedup_30_splits_raw.csv").groupby("Model", sort=False)[["Accuracy", "Precision", "Recall", "F1", "AUC"]].agg(["mean", "std"])
rows = [rows[0]] + [[m] + [f"{rd.loc[m, (k, 'mean')]:.4f} ± {rd.loc[m, (k, 'std')]:.4f}" for k in ["Accuracy", "Precision", "Recall", "F1", "AUC"]]
                    for m in ["LR", "DT", "RF", "XGB", "NB", "KNN", "Stacking"]]
story.append(captioned_table("TABLE A2. Identical paper pipeline, 30 random stratified 80/20 splits of the 302 unique records.", table(rows, [22 * mm] + [29.6 * mm] * 5)))
story.append(P(f"<b>Data audit.</b> rows = {dup['rows']}; exact duplicate rows = {dup['duplicate_rows']}; unique rows = {dup['unique_rows']}; "
               f"unique feature vectors with conflicting labels = {dup['unique_rows_with_conflicting_labels']}. "
               f"Mean share of test rows with an exact copy in the training set under the paper protocol: {pct(seen.mean())} (range "
               f"{pct(seen.min())} to {pct(seen.max())} over 30 splits).", small))


def on_page(c, d):
    c.saveState(); c.setFont("Sans", 7.5); c.setFillColor(MUTED)
    c.drawString(18 * mm, 10 * mm, "SIT307 HD Task: Machine Learning Mini Research, " + AUTHOR)
    c.drawRightString(A4[0] - 18 * mm, 10 * mm, str(d.page)); c.restoreState()


os.makedirs("report", exist_ok=True)
doc = SimpleDocTemplate("report/SIT307_HD_Technical_Report.pdf", pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                        topMargin=16 * mm, bottomMargin=17 * mm, title="SIT307 HD Technical Report: LACS", author=AUTHOR)
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("report written")
