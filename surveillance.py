import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import logit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from tabulate import tabulate
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

# ----------------------------
# Paths - CHANGE if needed
# ----------------------------
INPUT_PATH = r"C:\Users\everp\Documents\Surveillance\surveillance_recoded.csv"
OUT_DIR = r"C:\Users\everp\Documents\Surveillance\analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Utility functions
# ----------------------------
def pretty_print_df(df, title=None, max_rows=200):
    """Print dataframe in a human-friendly table form (tabulate)."""
    if title:
        print(f"\n=== {title} ===")
    if df.shape[0] == 0:
        print("[No rows]")
        return
    # limit size to avoid huge dumps
    if df.shape[0] > max_rows:
        display_df = df.head(max_rows)
        print("(showing first {} rows)".format(max_rows))
    else:
        display_df = df
    print(tabulate(display_df, headers="keys", tablefmt="psql", showindex=True))
    print()

def freq_pct(series, sort=True):
    """Return a DataFrame of counts and percentages for a Series."""
    vc = series.value_counts(dropna=False)
    pct = vc / vc.sum() * 100
    out = pd.DataFrame({'Count': vc, 'Percent': pct.round(2)})
    if sort:
        out = out.sort_values('Count', ascending=False)
    return out

def save_fig(fig, filename):
    path = os.path.join(OUT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {path}")

def try_ordered_model(endog, exog_df, exog_name):
    """Fit an ordered logistic (proportional odds) model if possible."""
    try:
        model = OrderedModel(endog, exog_df, distr='logit')
        res = model.fit(method='bfgs', disp=False)
        return res
    except Exception as e:
        print("OrderedModel failed:", e)
        return None

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(INPUT_PATH)
print(f"Loaded data with shape: {df.shape}\nColumns: {list(df.columns)}")

# ----------------------------
# Clean column names
# ----------------------------
def clean_columns(df):
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", " ", regex=True)  # collapse extra spaces
        .str.replace(r"[^\w\s]", "", regex=True)  # remove ? () ...
        .str.replace("  ", " ")
    )
    return df

df = clean_columns(df)

# ----------------------------
# Quick missingness report
# ----------------------------
miss = df.isnull().sum().sort_values(ascending=False)
miss_pct = (miss / len(df) * 100).round(2)
missing_report = pd.DataFrame({'MissingCount': miss, 'MissingPct': miss_pct})
pretty_print_df(missing_report, "Missingness report (variable-level)")

# ----------------------------
# Variables classification (based on your codebook)
# - binary_vars: 0/1 variables
# - nominal_vars: categories encoded as ints without inherent order (e.g., Current Position, Location)
# - ordinal_vars: ordered categories (e.g., Age Group (1-4), How familiar (1-3), Likert (1-5), Frequency)
# - multiple binary challenge_* variables already present
# ----------------------------
# You may adjust these lists if variable names differ slightly.
all_cols = df.columns.tolist()

# Heuristics to classify variables automatically where possible:
binary_prefixes = ['Do you work', 'Are you aware', 'Have you received', 'Has', 'Challenge_']
ordinal_likely = [
    'Age Group', "Years of Experience", "How familiar are you with the processes of ES (sewage sampling, lab testing)?",
    "How effective is ES in detecting poliovirus compared to  Acute Flaccid Paralysis (AFP) ?",
    "ES has played a role in polio eradication in Nigeria",
    "Frequency of environmental sample collection"
]
nominal_likely = ['Current Position', 'Location/State of Primary Assignment', 'Primary Role in Environmental Surveillance for Polio Eradication',
                  'Which system has faster virus detection?']

# Build lists by checking presence
binary_vars = [c for c in all_cols if any(c.startswith(p) for p in binary_prefixes)]
# Add explicit binary from codebook
binary_vars += [c for c in all_cols if c in [
    "Are you aware of environmental surveillance (ES) activities in your area?",
    "Have you received formal training on ES?",
    "Do you work with both AFP and ES systems?"
] and c not in binary_vars]

ordinal_vars = [c for c in all_cols if c in ordinal_likely]
nominal_vars = [c for c in all_cols if c in nominal_likely]

# Additional: role, gender etc. may be nominal but small categories; treat Gender as nominal
if 'Gender' in all_cols and 'Gender' not in nominal_vars:
    nominal_vars.append('Gender')

# Challenge binaries (match prefix)
challenge_cols = [c for c in all_cols if c.startswith('Challenge_')]

print("Variable groups detected:")
print(" - binary_vars:", binary_vars)
print(" - ordinal_vars:", ordinal_vars)
print(" - nominal_vars:", nominal_vars)
print(" - challenge binary cols (detected):", challenge_cols)
print()

# ----------------------------
# Clean column names for consistency
# ----------------------------
def sanitize_filename(name: str) -> str:
    """Sanitize a string to be a safe filename for Windows/Linux."""
    return re.sub(r'[<>:"/\\|?*]', "_", name)

# Overwrite save_fig to auto-sanitize
def save_fig(fig, filename):
    safe_filename = sanitize_filename(filename)
    path = os.path.join(OUT_DIR, safe_filename)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {path}")

# ----------------------------
# Label mapping (codebook)
# ----------------------------
label_maps = {
    "Gender": {1: "Male", 2: "Female"},
    "Age Group": {1: "18–30", 2: "31–40", 3: "41–50", 4: "51+"},
    "Current Position": {
        1: "Surveillance Officer",
        2: "Laboratory Scientist",
        3: "Environmental Health Officer",
        4: "Health Facility Worker",
        5: "Other"
    },
    "Primary Role in Environmental Surveillance for Polio Eradication": {
        1: "Sample collection",
        2: "Laboratory analysis",
        3: "Data management and reporting",
        4: "Supervision/Coordination",
        5: "Routine Inspection of Premises",
        6: "Other"
    },
    "Years of Experience": {
        1: "Less than 5 years",
        2: "5–10 years",
        3: "11–15 years",
        4: "Over 15 years"
    },
    "Are you aware of environmental surveillance (ES) activities in your area?": {
        1: "Yes", 0: "No"
    },
    "How familiar are you with the processes of ES (sewage sampling, lab testing)?": {
        1: "Very familiar",
        2: "Somewhat familiar",
        3: "Not familiar"
    },
    "What is environmental surveillance about?": {
        1: "Tracking wildlife populations",
        2: "Monitoring environmental hazards",
        3: "Enforcing workplace safety",
        4: "Studying genetic diseases"
    },
    "Have you received formal training on ES?": {1: "Yes", 0: "No"},
    "How effective is ES in detecting poliovirus compared to  Acute Flaccid Paralysis (AFP) ?": {
        1: "More effective",
        2: "Equally effective",
        3: "Less effective",
        0: "Don’t know"
    },
    "Has ES in your area contributed to early detection of poliovirus cases?": {
        1: "Yes", 0: "No", 2: "Not sure"
    },
    "ES has played a role in polio eradication in Nigeria": {
        1: "Strongly disagree",
        2: "Disagree",
        3: "Neutral",
        4: "Agree",
        5: "Strongly agree"
    },
    "Frequency of environmental sample collection": {
        1: "Weekly",
        2: "Bi-weekly",
        3: "Monthly",
        4: "Rarely",
        0: "Not applicable"
    },
    "Do you work with both AFP and ES systems?": {1: "Yes", 0: "No"},
    "Which system has faster virus detection?": {
        1: "Environmental Surveillance",
        2: "Acute Flaccid Paralysis",
        3: "Both",
        4: "Not sure"
    },
    # Challenges (binary)
    "Challenge_Poor_funding": {1: "Yes", 0: "No"},
    "Challenge_Inadequate_training": {1: "Yes", 0: "No"},
    "Challenge_Sample_collection_difficulties": {1: "Yes", 0: "No"},
    "Challenge_Delay_in_lab_analysis": {1: "Yes", 0: "No"},
    "Challenge_Insecurity_in_sampling_areas": {1: "Yes", 0: "No"},
    "Challenge_Logistical/transport_issues": {1: "Yes", 0: "No"},
    "Challenge_Lack_of_community_awareness": {1: "Yes", 0: "No"},
    "Challenge_Other…": {1: "Yes", 0: "No"}
}

def apply_labels(df):
    """Map numeric codes to human-readable labels using codebook."""
    for col, mapping in label_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(df[col])
    return df

df = apply_labels(df)

# ----------------------------
# Plotting utility for nominal/ordinal vars
# ----------------------------
import textwrap

def plot_bar_counts(series, title, filename, rotate_xticks=False):
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = series.value_counts(dropna=False)
    counts.plot(kind="bar", ax=ax, color="skyblue", edgecolor="black")

    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=12)

    # Format x-tick labels
    labels = [textwrap.fill(str(lab), 15) for lab in counts.index]
    ax.set_xticklabels(labels, rotation=45 if rotate_xticks else 0, ha="right")

    # Add counts + percentages on bars
    total = counts.sum()
    for p, val in zip(ax.patches, counts.values):
        ax.annotate(f"{val}\n({val/total:.1%})",
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha="center", va="bottom", fontsize=9)

    save_fig(fig, filename)

# ----------------------------
# DESCRIPTIVE STATISTICS
# ----------------------------
print("\n##### DESCRIPTIVE STATISTICS: COUNTS & PERCENTAGES #####")

# For each categorical (nominal or ordinal) variable: counts & percentages table
cat_vars = sorted(list(set(nominal_vars + ordinal_vars + binary_vars + challenge_cols)))
for col in cat_vars:
    if col not in df.columns:
        continue
    table = freq_pct(df[col])
    pretty_print_df(table, f"Frequency: {col}")

# For numeric-looking variables, show summary stats
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# exclude code-coded categorical columns already shown; keep to show min/median/max
numeric_cols = [c for c in numeric_cols if c not in cat_vars]
if numeric_cols:
    print("Numeric summary for numeric columns (not categorical):")
    pretty_print_df(df[numeric_cols].describe().T)

# ----------------------------
# VISUALIZATIONS (single-plot per figure)
# - Bar charts for categorical variables (counts)
# - Horizontal bar charts for longer categories (like states)
# - Histogram for any numeric variables (few)
# - Stacked bar example: Primary Role by Gender
# ----------------------------
print("\n##### CREATING PLOTS #####")

# Set a palette
import seaborn as sns
palette = sns.color_palette("deep")

# Function to plot bar chart for a categorical variable
def plot_bar_counts(series, title, filename, rotate_xticks=False, horizontal=False, max_categories=30):
    vc = series.value_counts(dropna=False)
    if len(vc) > max_categories and not horizontal:
        # show horizontal for many categories
        horizontal = True
    fig, ax = plt.subplots(figsize=(10,6))
    if horizontal:
        vc.sort_values().plot(kind='barh', ax=ax)
        ax.set_xlabel("Count")
    else:
        vc.plot(kind='bar', ax=ax)
        ax.set_ylabel("Count")
        if rotate_xticks:
            plt.xticks(rotation=45, ha='right')
    ax.set_title(title)
    # annotate counts on bars
    for p in ax.patches:
        width = p.get_width() if horizontal else p.get_height()
        if np.isnan(width):
            continue
        if horizontal:
            ax.text(p.get_width() + max(vc) * 0.01, p.get_y() + p.get_height()/2,
                    int(p.get_width()), va='center')
        else:
            ax.text(p.get_x() + p.get_width()/2, p.get_height() + max(vc)*0.01,
                    int(p.get_height()), ha='center')
    save_fig(fig, filename)

# Plot each categorical variable
for col in cat_vars:
    if col not in df.columns:
        continue
    # For location/state, use horizontal due to many categories
    horizontal = True if 'Location/State' in col else False
    fname = f"bar_{col.replace(' ','_').replace('/','_')}.png"
    plot_bar_counts(df[col], f"Counts of {col}", fname, rotate_xticks=True, horizontal=horizontal, max_categories=40)

# Example stacked bar: Primary Role by Gender (if both exist)
if 'Primary Role in Environmental Surveillance for Polio Eradication' in df.columns and 'Gender' in df.columns:
    cross = pd.crosstab(df['Primary Role in Environmental Surveillance for Polio Eradication'],
                        df['Gender'], normalize='index') * 100
    pretty_print_df(cross.round(2), "Primary Role (%) by Gender (row %)")
    # Plot stacked bar (percent)
    fig, ax = plt.subplots(figsize=(10,6))
    cross.plot(kind='bar', stacked=True, ax=ax)
    ax.set_ylabel("Percent (row%)")
    ax.set_title("Primary Role by Gender (row percent)")
    ax.legend(title='Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
    save_fig(fig, "stacked_PrimaryRole_by_Gender.png")

# Correlation heatmap for numeric-coded variables (use spearman for ordinal)
num_for_corr = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if df[c].nunique()>1]
if len(num_for_corr) >= 2:
    corr = df[num_for_corr].corr(method='spearman')
    fig, ax = plt.subplots(figsize=(12,10))
    im = ax.imshow(corr, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index)
    ax.set_title("Spearman correlation (coded variables)")
    # annotate
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha='center', va='center', color='w' if abs(corr.iloc[i,j])>0.5 else 'k')
    fig.colorbar(im, ax=ax, fraction=0.02)
    save_fig(fig, "spearman_correlation_heatmap.png")

print("\nPlots created.\n")

# ----------------------------
# INFERENTIAL STATISTICS
# - 1) Chi-square tests between pairs of categorical variables of interest.
# - 2) For ordinal vs group: Kruskal-Wallis or Mann-Whitney.
# - 3) Logistic regression predicting 'Are you aware...' (binary) from a small set of predictors.
# - 4) Ordered logistic for 'How familiar...' (ordinal) if possible.
# ----------------------------
print("\n##### INFERENTIAL STATISTICS #####")

def chi2_test(a, b, colA, colB, fisher_threshold=5):
    """Run chi-square or Fisher exact depending on table size."""
    table = pd.crosstab(a, b)
    pretty_print_df(table, f"Contingency table: {colA} x {colB}")
    # If 2x2 and small expected values -> Fisher exact
    if table.size == 4:
        # compute expected counts
        chi2, p, dof, exp = stats.chi2_contingency(table, correction=False)
        if (exp < fisher_threshold).any():
            # Fisher exact requires 2x2; ensure convertible to int
            try:
                oddsr, p_f = stats.fisher_exact(table.values)
                print(f"Fisher exact test (2x2): p = {p_f:.4f}")
                return ('fisher', p_f)
            except Exception:
                print("Fisher exact failed; falling back to chi2")
        # else use chi2
    chi2, p, dof, exp = stats.chi2_contingency(table)
    print(f"Chi-square: chi2 = {chi2:.3f}, dof = {dof}, p = {p:.4f}")
    # display expected if needed
    pretty_print_df(pd.DataFrame(exp, index=table.index, columns=table.columns), "Expected counts")
    return ('chi2', p)

# a) Chi-square examples: Gender vs Awareness; Primary Role vs Awareness
pairs_to_test = []
if 'Gender' in df.columns and 'Are you aware of environmental surveillance (ES) activities in your area?' in df.columns:
    pairs_to_test.append(('Gender', 'Are you aware of environmental surveillance (ES) activities in your area?'))
if 'Primary Role in Environmental Surveillance for Polio Eradication' in df.columns and 'Are you aware of environmental surveillance (ES) activities in your area?' in df.columns:
    pairs_to_test.append(('Primary Role in Environmental Surveillance for Polio Eradication', 'Are you aware of environmental surveillance (ES) activities in your area?'))
if 'Location/State of Primary Assignment' in df.columns and 'Are you aware of environmental surveillance (ES) activities in your area?' in df.columns:
    pairs_to_test.append(('Location/State of Primary Assignment', 'Are you aware of environmental surveillance (ES) activities in your area?'))

for a_col, b_col in pairs_to_test:
    print(f"\n--- Association test: {a_col} vs {b_col} ---")
    chi2_test(df[a_col], df[b_col], a_col, b_col)

# b) Ordinal comparisons: e.g., "How familiar" by Primary Role --> Kruskal-Wallis
if ("How familiar are you with the processes of ES (sewage sampling, lab testing)?" in df.columns
    and "Primary Role in Environmental Surveillance for Polio Eradication" in df.columns):
    var = "How familiar are you with the processes of ES (sewage sampling, lab testing)?"
    group = "Primary Role in Environmental Surveillance for Polio Eradication"
    print(f"\n--- Kruskal-Wallis: {var} by {group} ---")
    groups = []
    labels = []
    for lvl, sub in df.groupby(group):
        arr = sub[var].dropna().values
        if len(arr) >= 3:
            groups.append(arr)
            labels.append(str(lvl))
    if len(groups) >= 2:
        kw = stats.kruskal(*groups)
        print(f"Kruskal-Wallis H = {kw.statistic:.3f}, p = {kw.pvalue:.4f}")
        # If significant, show pairwise Mann-Whitney with Bonferroni
        if kw.pvalue < 0.05:
            print("Post-hoc pairwise Mann-Whitney tests (Bonferroni-corrected):")
            from itertools import combinations
            results = []
            for (lvl1, lvl2) in combinations(df[group].unique(), 2):
                a = df.loc[df[group]==lvl1, var].dropna()
                b = df.loc[df[group]==lvl2, var].dropna()
                if len(a)>=3 and len(b)>=3:
                    stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                    results.append((lvl1, lvl2, stat, p))
            # Bonferroni
            m = len(results)
            for r in results:
                print(f"{r[0]} vs {r[1]}: U = {r[2]:.3f}, p_uncorrected = {r[3]:.4f}, p_bonf = {min(r[3]*m,1.0):.4f}")

# c) Logistic regression: predict 'Are you aware...' (binary) using Gender, Age Group, Primary Role, Years Experience
bin_outcome = "Are you aware of environmental surveillance (ES) activities in your area?"
if bin_outcome in df.columns:
    print(f"\n--- Logistic regression predicting {bin_outcome} ---")
    # Prepare predictors: choose sensible predictors if present
    predictors = []
    for cand in ['Gender', 'Age Group', 'Primary Role in Environmental Surveillance for Polio Eradication', 'Years of Experience']:
        if cand in df.columns:
            predictors.append(cand)
    if len(predictors) == 0:
        print("No predictors available for logistic regression.")
    else:
        # Build formula after one-hot encoding categorical predictors (except numeric ordinal ones)
        # We'll convert categorical (nominal) to dummies, keep ordinal numeric as-is
        df_model = df[[bin_outcome] + predictors].dropna()
        y = df_model[bin_outcome]
        X = pd.DataFrame(index=df_model.index)
        for p in predictors:
            # treat these as categorical if they have small integer codes and represent categories
            if df_model[p].nunique() <= 8 and p in nominal_vars:
                d = pd.get_dummies(df_model[p].astype(str), prefix=p, drop_first=True)
                X = pd.concat([X, d], axis=1)
            else:
                # treat as numeric (ordinal)
                X[p] = df_model[p]
        X = sm.add_constant(X, has_constant='add')
        try:
            model = sm.Logit(y, X).fit(disp=False)
            print(model.summary())
            # Save model summary to file
            with open(os.path.join(OUT_DIR, "logistic_awareness_summary.txt"), "w") as f:
                f.write(model.summary().as_text())
            print("Logistic regression summary saved.")
        except Exception as e:
            print("Logit model failed:", e)

# d) Ordered logistic for "How familiar..." (1-3) if available
ord_var = "How familiar are you with the processes of ES (sewage sampling, lab testing)?"
if ord_var in df.columns:
    print(f"\n--- Ordered logistic regression predicting {ord_var} ---")
    # predictors: Age Group, Gender, Primary Role
    preds = [p for p in ['Gender', 'Age Group', 'Primary Role in Environmental Surveillance for Polio Eradication'] if p in df.columns]
    if preds:
        df_om = df[[ord_var] + preds].dropna()
        # endog must be 1...k integers
        endog = df_om[ord_var].astype(int)
        # create exog DataFrame with dummies for nominal preds, keep ordinal as numeric
        exog = pd.DataFrame(index=df_om.index)
        for p in preds:
            if p in nominal_vars and df_om[p].nunique() <= 12:
                exog = pd.concat([exog, pd.get_dummies(df_om[p].astype(str), prefix=p, drop_first=True)], axis=1)
            else:
                exog[p] = df_om[p]
        # add constant
        exog = sm.add_constant(exog, has_constant='add')
        om_res = try_ordered_model(endog, exog, preds)
        if om_res is not None:
            print(om_res.summary())
            with open(os.path.join(OUT_DIR, "orderedlogit_familiarity_summary.txt"), "w") as f:
                f.write(om_res.summary().as_text())
            print("Ordered logistic summary saved.")
    else:
        print("No predictors available for ordered logistic.")

# e) Report a few cross-tabs of academic interest: Primary Role x Which system has faster detection, Role x Frequency
interest_pairs = [
    ('Primary Role in Environmental Surveillance for Polio Eradication', 'Which system has faster virus detection?'),
    ('Primary Role in Environmental Surveillance for Polio Eradication', 'Frequency of environmental sample collection')
]
for a_col, b_col in interest_pairs:
    if a_col in df.columns and b_col in df.columns:
        print(f"\n--- Cross-tab: {a_col} x {b_col} ---")
        table = pd.crosstab(df[a_col], df[b_col], normalize='index') * 100
        pretty_print_df(table.round(2), f"{a_col} (%) by {b_col} (row %)")
        # Chi-square test for association
        chi2_test(df[a_col], df[b_col], a_col, b_col)

# ----------------------------
# FINAL: Export cleaned table summaries and model outputs
# ----------------------------
# Save frequency tables for all categorical variables into CSVs in OUT_DIR
freq_folder = os.path.join(OUT_DIR, "freq_tables")
os.makedirs(freq_folder, exist_ok=True)
for col in cat_vars:
    if col in df.columns:
        freq = freq_pct(df[col]).reset_index().rename(columns={'index': col})
        freq.to_csv(os.path.join(freq_folder, f"freq_{col.replace(' ','_').replace('/','_')}.csv"), index=False)

print(f"\nFrequency tables saved to: {freq_folder}")
print(f"All outputs saved to: {OUT_DIR}")

print("\nANALYSIS COMPLETE. Read the files in the output directory for plots and model summaries.")
