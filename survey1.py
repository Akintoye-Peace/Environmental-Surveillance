import os
import re
import math
import textwrap
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

# --------------------------
# SETTINGS & PATHS
# --------------------------
CSV_PATH = r"C:\Users\everp\Documents\Surveillance\surveillance_recoded.csv"
OUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
TEXT_DIR = os.path.join(OUT_DIR, "text")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# Excel outputs
DESCRIPTIVE_XLSX = os.path.join(OUT_DIR, "descriptive_tables.xlsx")
INFERENTIAL_XLSX = os.path.join(OUT_DIR, "inferential_tables.xlsx")
MODELS_XLSX = os.path.join(OUT_DIR, "model_results.xlsx")

# --------------------------
# CODEBOOK (numeric -> label) exactly as you provided
# --------------------------
CODEBOOK = {
    "Gender": {1: "Male", 2: "Female"},
    "Age Group": {1: "18-'30", 2: "31-'40", 3: "41-'50", 4: "51+"},
    "Current Position": {
        1: "Surveillance Officer",
        2: "Laboratory Scientist",
        3: "Environmental Health Officer",
        4: "Health Facility Worker",
        5: "Other"
    },
    "Years of Experience": {
        1: "Less than 5 years",
        2: "5-'10 years",
        3: "11-'15 years",
        4: "Over 15 years"
    },
    "Are you aware of environmental surveillance (ES) activities in your area?": {1: "Yes", 0: "No"},
    "How familiar are you with the processes of ES (sewage sampling, lab testing)?": {
        3: "Very familiar",
        2: "Somewhat familiar",
        1: "Not familiar"
    },
    "What is environmental surveillance about?": {
        1: "Tracking wildlife populations to maintain ecological balance and biodiversity",
        2: "Monitoring and analyzing environmental hazards (e.g., pollution, toxins) to prevent health risks and guide public health actions",
        3: "Enforcing workplace safety regulations to prevent occupational injuries",
        4: "Studying genetic diseases to develop personalized medical treatments"
    },
    "Have you received formal training on ES?": {1: "Yes", 0: "No"},
    "How effective is ES in detecting poliovirus compared to  Acute Flaccid Paralysis (AFP) ?": {
        3: "More effective",
        2: "Equally effective",
        1: "Less effective",
        0: "Don't know"
    },
    "Has ES in your area contributed to early detection of poliovirus cases?": {1: "Yes", 0: "No", 2: "Not sure"},
    "ES has played a role in polio eradication in Nigeria": {
        5: "Strongly agree", 4: "Agree", 3: "Neutral", 2: "Disagree", 1: "Strongly disagree"
    },
    "Frequency of environmental sample collection": {
        4: "Weekly", 3: "Bi-weekly", 2: "Monthly", 1: "Rarely", 0: "Not applicable"
    },
    "Do you work with both AFP and ES systems?": {1: "Yes", 0: "No"},
    "Primary Role in Environmental Surveillance for Polio Eradication": {
        1: "Routine Inspection of Premises",
        2: "Sample collection",
        3: "Supervision/Coordination",
        4: "Data management and reporting",
        5: "Nurse Officer",
        6: "Laboratory analysis"
    },
    "Which system has faster virus detection?": {
        1: "Acute Flaccid Paralysis (AFP)",
        2: "Environmental Surveillance (ES)",
        3: "Both",
        4: "Not sure"
    }
}

# Challenges: final binary column names to create (if parsing multi-select); keep canonical labels
CHALLENGE_OPTIONS = [
    "Poor funding",
    "Inadequate training",
    "Sample collection difficulties",
    "Delay in lab analysis",
    "Insecurity in sampling areas",
    "Logistical/transport issues",
    "Lack of community awareness",
    "Other"
]

# --------------------------
# HELPERS: Normalization & fuzzy column finder
# --------------------------
_punct_re = re.compile(r"[^\w\s]")

def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.strip().lower()
    s = _punct_re.sub(" ", s)      # remove punctuation
    s = re.sub(r"\s+", " ", s)     # collapse spaces
    return s

def find_column(df, pattern_keywords):
    """
    pattern_keywords: string or list of keyword options
    returns the first column name that matches (case/space/punct-insensitive), or None
    """
    if isinstance(pattern_keywords, str):
        patterns = [pattern_keywords]
    else:
        patterns = pattern_keywords
    col_norm = {c: normalize_text(c) for c in df.columns}
    for pat in patterns:
        pat_n = normalize_text(pat)
        for col, col_n in col_norm.items():
            if pat_n in col_n:
                return col
    # fallback: try any column that contains all words from pat
    for pat in patterns:
        words = [w for w in normalize_text(pat).split() if len(w) > 2]
        for col, col_n in col_norm.items():
            if all(w in col_n for w in words):
                return col
    return None

# --------------------------
# LOAD DATA (robust)
# --------------------------
print("\nLoading data from:", CSV_PATH)
df_raw = pd.read_csv(CSV_PATH, dtype=str)  # read all as strings initially
print("Raw data shape:", df_raw.shape)

# Clean column names (strip)
df_raw.columns = [c.strip() for c in df_raw.columns]

# Trim string cells
df_raw = df_raw.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# --------------------------
# MAP variables: create *_label and *_code columns
# Works both when dataset has numeric codes or textual labels.
# --------------------------
df = df_raw.copy()

mapping_info = {}  # store found columns mapping for reporting

for cb_col, code_map in CODEBOOK.items():
    found = find_column(df, cb_col)
    if not found:
        # Try shorter name match (first few words)
        short = " ".join(cb_col.split()[:4])
        found = find_column(df, short)
    if not found:
        print(f"WARNING: Could not find column matching '{cb_col}' in dataset. Skipping mapping for this variable.")
        continue
    # Build inverse map label -> code
    inv_map = {v.lower(): k for k, v in code_map.items()}
    label_col = found + "_label"
    code_col = found + "_code"
    mapping_info[cb_col] = {"dataset_column": found, "label_col": label_col, "code_col": code_col}
    labels = []
    codes = []
    for v in df[found].astype(str).fillna("").values:
        v_str = v.strip()
        if v_str == "" or v_str.lower() in ["nan", "none"]:
            labels.append(np.nan); codes.append(np.nan); continue
        # if purely numeric (like "1" or "2")
        if re.fullmatch(r"-?\d+", v_str):
            num = int(v_str)
            if num in code_map:
                labels.append(code_map[num])
                codes.append(num)
            else:
                # numeric but not in map
                labels.append(v_str); codes.append(np.nan)
            continue
        # not numeric: check if it matches a label in inverse map
        v_norm = v_str.lower()
        if v_norm in inv_map:
            code = inv_map[v_norm]
            labels.append(code_map[code])
            codes.append(code)
            continue
        # sometimes the dataset already contains the short label but with different capitalization
        match_found = False
        for code_k, lab in code_map.items():
            if normalize_text(lab) == normalize_text(v_str):
                labels.append(lab); codes.append(code_k); match_found = True; break
        if match_found:
            continue
        # final fallback: if value equals numeric code with trailing non-digit characters (rare)
        digits = re.findall(r"\d+", v_str)
        if digits:
            num = int(digits[0])
            if num in code_map:
                labels.append(code_map[num]); codes.append(num)
                continue
        # otherwise keep original string label and NaN code
        labels.append(v_str); codes.append(np.nan)
    df[label_col] = labels
    df[code_col] = codes

# States column: find location/state column and standardize name
loc_col = find_column(df, ["Location/State of Primary Assignment", "Location", "State of Primary Assignment"])
if loc_col:
    df["state_label"] = df[loc_col].astype(str).replace({"": np.nan})
else:
    print("WARNING: No state/location column found. State-by-state analyses will be skipped.")
    df["state_label"] = np.nan

# Challenges: if dataset already has binary challenge cols (start with 'Challenge_' or 'Binary Challenge')
challenge_bin_cols = [c for c in df.columns if normalize_text(c).startswith("challenge_") or normalize_text(c).startswith("binary challenge")]
if challenge_bin_cols:
    # Ensure binary numeric dtype
    for c in challenge_bin_cols:
        df[c] = pd.to_numeric(df[c].replace({"": np.nan, "nan": np.nan}), errors='coerce').fillna(0).astype(int)
else:
    # try to find a raw multi-select column and parse it
    chal_col = find_column(df, "Challenges encountered in ES")
    if chal_col:
        # create binary columns
        for opt in CHALLENGE_OPTIONS:
            col_name = "Challenge_" + re.sub(r"[^\w]", "_", opt).strip("_")
            df[col_name] = 0
        # parse each row
        for idx, val in df[chal_col].astype(str).fillna("").items():
            if val == "" or val.lower() in ["nan", "none"]:
                continue
            parts = re.split(r"[;,/|]", val)
            parts = [p.strip().lower() for p in parts if p.strip() != ""]
            for opt in CHALLENGE_OPTIONS:
                if any(opt.lower() in p for p in parts):
                    col_name = "Challenge_" + re.sub(r"[^\w]", "_", opt).strip("_")
                    df.at[idx, col_name] = 1
    else:
        print("No challenges column found (binary or multi-select). Skipping challenge parsing.")

# --------------------------
# Utility: pretty freq table (counts + percent) and printing to console
# --------------------------
def freq_table(series, name=None, dropna=False):
    name = name or series.name
    vc = series.value_counts(dropna=dropna)
    total = vc.sum()
    pct = (vc / total * 100).round(2)
    table = pd.DataFrame({"Count": vc, "Percent": pct})
    return table

def print_and_save_dataframe(df_tab, title=None):
    if title:
        print("\n" + "="*100)
        print(f"{title}")
        print(df_tab.to_string())
        print("="*100 + "\n")
    else:
        print(df_tab.to_string())

# --------------------------
# DESCRIPTIVE ANALYSIS: overall + by-state
# Save all descriptive tables into an Excel workbook
# --------------------------
print("\nRUNNING DESCRIPTIVE ANALYSIS...\n")
desc_writer = pd.ExcelWriter(DESCRIPTIVE_XLSX, engine="openpyxl")

# 1) For every CODEBOOK variable, print counts (use label col where available)
for cb_col in CODEBOOK.keys():
    mapping = mapping_info.get(cb_col)
    if mapping:
        lab_col = mapping["label_col"]
        tab = freq_table(df[lab_col].fillna("Missing"))
        print_and_save_dataframe(tab, title=f"Distribution: {cb_col} (using '{mapping['dataset_column']}')")
        tab.to_excel(desc_writer, sheet_name=re.sub(r"[^\w]", "_", cb_col)[:31])
    else:
        # attempt to find a matching raw column and show counts
        found = find_column(df_raw, cb_col)
        if found:
            tab = freq_table(df_raw[found].fillna("Missing"))
            print_and_save_dataframe(tab, title=f"Distribution (raw): {found}")
            tab.to_excel(desc_writer, sheet_name=re.sub(r"[^\w]", "_", found)[:31])
        else:
            print(f"NOTICE: column for '{cb_col}' not found; skipped descriptive table.")

# 2) State by state: awareness, training, ES contribution, familiarity
if df["state_label"].notna().sum() > 0:
    states = df["state_label"].dropna().unique().tolist()
    # awareness by state
    awareness_col = mapping_info.get("Are you aware of environmental surveillance (ES) activities in your area?")
    training_col = mapping_info.get("Have you received formal training on ES?")
    contributed_col = mapping_info.get("Has ES in your area contributed to early detection of poliovirus cases?")
    familiarity_col = mapping_info.get("How familiar are you with the processes of ES (sewage sampling, lab testing)?")
    for metric_name, mapping in [("Awareness", awareness_col), ("Training", training_col),
                                ("ES_contributed", contributed_col), ("Familiarity", familiarity_col)]:
        if mapping is None:
            print(f"NOTE: {metric_name} column not found; skipping state-by-state for this metric.")
            continue
        lab_col = mapping["label_col"]
        rows = []
        for s in sorted(df["state_label"].dropna().unique()):
            sub = df[df["state_label"] == s]
            n = len(sub)
            if n == 0:
                continue
            counts = sub[lab_col].value_counts(dropna=False)
            # Percent 'Yes' if binary yes/no metric; else show distribution
            rows.append((s, n, counts.to_dict()))
        # For Excel: save per-state distribution (wide)
        per_state_table = pd.crosstab(df["state_label"], df[lab_col], dropna=False)
        print(f"\nState-by-state distribution for {metric_name} (rows=state, cols=responses):")
        print(per_state_table)
        per_state_table.to_excel(desc_writer, sheet_name=f"{metric_name}_by_state"[:31])
else:
    print("State-by-state analyses skipped (no state data).")

# 3) Challenges summary (binary columns)
chal_cols = [c for c in df.columns if normalize_text(c).startswith("challenge_")]
if chal_cols:
    chal_counts = df[chal_cols].sum().sort_values(ascending=False)
    chal_df = pd.DataFrame({"Count": chal_counts, "Percent": (chal_counts / len(df) * 100).round(2)})
    print("\nOperational challenges (counts & percent):")
    print(chal_df)
    chal_df.to_excel(desc_writer, sheet_name="Challenges_Summary")
else:
    print("No challenge binary columns found; skipping challenge summary.")

# Save cleaned labelled dataset too
cleaned_csv = os.path.join(OUT_DIR, "cleaned_labelled_data.csv")
df.to_csv(cleaned_csv, index=False)
df.to_excel(desc_writer, sheet_name="cleaned_labelled_data", index=False)

desc_writer.save()
print(f"\nDescriptive tables saved to: {DESCRIPTIVE_XLSX}")
print(f"Cleaned labelled dataset saved to: {cleaned_csv}")

# --------------------------
# PLOTS: each plot saved individually and labelled using label columns
# (each plot stands alone, no combined multi-panel)
# --------------------------
print("\nCREATING PLOTS...")

sns.set(style="whitegrid")

def save_barplot_counts(series, title, filename, horizontal=False):
    counts = series.value_counts(dropna=False)
    labels = [str(x) for x in counts.index]
    values = counts.values
    plt.figure(figsize=(8,5))
    if horizontal:
        sns.barplot(x=values, y=labels)
        plt.xlabel("Count")
        plt.ylabel("")
    else:
        sns.barplot(x=labels, y=values)
        plt.ylabel("Count")
        plt.xlabel("")
    plt.title(title)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, filename)
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved plot:", out)

# Example plots for key items
for cb_col in ["Gender", "Age Group", "Current Position", "How familiar are you with the processes of ES (sewage sampling, lab testing)?",
               "What is environmental surveillance about?", "How effective is ES in detecting poliovirus compared to  Acute Flaccid Paralysis (AFP) ?",
               "Has ES in your area contributed to early detection of poliovirus cases?", "Which system has faster virus detection?",
               "Frequency of environmental sample collection"]:
    mapping = mapping_info.get(cb_col)
    if mapping:
        lab_col = mapping["label_col"]
        safe_name = re.sub(r"[^\w]", "_", cb_col)[:40]
        save_barplot_counts(df[lab_col].fillna("Missing"), title=cb_col, filename=f"{safe_name}.png", horizontal=True)
    else:
        # try to find raw col
        found = find_column(df_raw, cb_col)
        if found:
            save_barplot_counts(df_raw[found].fillna("Missing"), title=found, filename=f"{re.sub(r'[^\\w]','_',found)[:40]}.png", horizontal=True)

# plot challenges as horizontal bar
if chal_cols:
    chal_sums = df[chal_cols].sum().sort_values(ascending=True)
    plt.figure(figsize=(8,6))
    sns.barplot(x=chal_sums.values, y=[c.replace("Challenge_", "") for c in chal_sums.index])
    plt.xlabel("Number of respondents selecting challenge")
    plt.ylabel("")
    plt.title("Challenges encountered in ES (multi-select totals)")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "challenges_counts.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved plot:", out)

# state awareness bar (if state exists)
if df["state_label"].notna().sum() > 0 and mapping_info.get("Are you aware of environmental surveillance (ES) activities in your area?"):
    aware_lab = mapping_info["Are you aware of environmental surveillance (ES) activities in your area?"]["label_col"]
    state_awareness = pd.crosstab(df["state_label"], df[aware_lab], normalize='index').fillna(0)
    # Plot percent aware by state (sorted)
    aware_pct = state_awareness.get("Yes", pd.Series(0, index=state_awareness.index))
    aware_pct = aware_pct.sort_values(ascending=False)
    plt.figure(figsize=(10,6))
    sns.barplot(x=aware_pct.values*100, y=aware_pct.index)
    plt.xlabel("Percent aware (%)")
    plt.title("Percent aware of ES by state (sorted)")
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "state_awareness_pct.png")
    plt.savefig(out, dpi=300)
    plt.close()
    print("Saved plot:", out)

print("All plots saved under:", PLOTS_DIR)

# --------------------------
# INFERENTIAL ANALYSIS
# Save contingency tables and test results to Excel
# --------------------------
print("\nRUNNING INFERENTIAL ANALYSES...")

inf_writer = pd.ExcelWriter(INFERENTIAL_XLSX, engine="openpyxl")

def chi2_test_table(col1, col2, df_local, sheetname):
    ct = pd.crosstab(df_local[col1].fillna("Missing"), df_local[col2].fillna("Missing"))
    try:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        result = pd.DataFrame({
            "chi2": [chi2], "p-value": [p], "dof": [dof]
        })
    except Exception as e:
        result = pd.DataFrame({"error":[str(e)]})
    # write ct and result to sheet: we will write ct first then result in next sheet with suffix
    ct.to_excel(inf_writer, sheet_name=sheetname[:31])
    result.to_excel(inf_writer, sheet_name=(sheetname[:27] + "_res")[:31], index=False)
    print(f"\nChi-square test: {col1} vs {col2}")
    print(ct)
    print(result)
    return ct, result

# Objective 1: Contribution of ES to early detection
es_contrib_map = mapping_info.get("Has ES in your area contributed to early detection of poliovirus cases?")
if es_contrib_map:
    es_label = es_contrib_map["label_col"]
    overall_tab = freq_table(df[es_label].fillna("Missing"))
    print_and_save_dataframe(overall_tab, title="Overall: Has ES contributed to early detection?")
    overall_tab.to_excel(inf_writer, sheet_name="ES_contrib_overall")
    # Associations: ES_contributed vs awareness, training, familiarity, primary role, works both
    assoc_vars = [
        "Are you aware of environmental surveillance (ES) activities in your area?",
        "Have you received formal training on ES?",
        "How familiar are you with the processes of ES (sewage sampling, lab testing)?",
        "Primary Role in Environmental Surveillance for Polio Eradication",
        "Do you work with both AFP and ES systems?"
    ]
    for var in assoc_vars:
        mapping = mapping_info.get(var)
        if mapping:
            ct, res = chi2_test_table(mapping["label_col"], es_label, df, sheetname=f"{var}_vs_EScontrib")
        else:
            print(f"Assoc variable '{var}' not found in dataset, skipping.")

else:
    print("ES-contributed column not found; cannot run Objective 1 inferential tests.")

# Objective 2: Compare ES vs AFP in selected states
# We'll use the "How effective is ES in detecting poliovirus compared to AFP?" variable and "Which system has faster virus detection?".
effect_map = mapping_info.get("How effective is ES in detecting poliovirus compared to  Acute Flaccid Paralysis (AFP) ?")
which_map = mapping_info.get("Which system has faster virus detection?")
if effect_map and df["state_label"].notna().sum() > 0:
    # choose selected high-risk states as top 6 by response counts (proxy)
    top_states = df["state_label"].value_counts().nlargest(6).index.tolist()
    print("\nSelected states for state-wise comparison (top 6 by responses):", top_states)
    # For each selected state, show distribution of 'effectiveness' and 'which system faster'
    per_state_effect = pd.crosstab(df["state_label"], df[effect_map["label_col"]], normalize='index').loc[top_states].fillna(0)
    print("\nEffectiveness (percent) by selected states (rows=state):")
    print((per_state_effect * 100).round(2))
    per_state_effect.to_excel(inf_writer, sheet_name="Effectiveness_by_state")
    if which_map:
        per_state_faster = pd.crosstab(df["state_label"], df[which_map["label_col"]], normalize='index').loc[top_states].fillna(0)
        per_state_faster.to_excel(inf_writer, sheet_name="WhichFaster_by_state")
        print("\nWhich system faster (percent) by selected states:")
        print((per_state_faster * 100).round(2))
    # Chi-square across selected states for 'More effective' vs others
    # Build contingency table (state x effectiveness label)
    ct = pd.crosstab(df[df["state_label"].isin(top_states)]["state_label"], df[effect_map["label_col"]])
    try:
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        res = pd.DataFrame({"chi2": [chi2], "p": [p], "dof": [dof]})
    except Exception as e:
        res = pd.DataFrame({"error":[str(e)]})
    ct.to_excel(inf_writer, sheet_name="state_effectiveness_ct")
    res.to_excel(inf_writer, sheet_name="state_effectiveness_test")
    print("\nChi-square for effectiveness across selected states:\n", res)
else:
    print("Effectiveness or state data missing; skipping Objective 2 state comparisons.")

# Objective 3: Operational challenges -> association with ES contribution
if chal_cols and es_contrib_map:
    # For each challenge do chi-square vs es_contrib
    chal_results = []
    for c in chal_cols:
        ct = pd.crosstab(df[c], df[es_label])
        try:
            chi2, p, dof, exp = stats.chi2_contingency(ct)
        except Exception as e:
            chi2, p, dof = (np.nan, np.nan, np.nan)
        chal_results.append({"challenge": c, "chi2": chi2, "p": p, "dof": dof})
        ct.to_excel(inf_writer, sheet_name=(c[:27] + "_ct")[:31])
    chal_res_df = pd.DataFrame(chal_results)
    print("\nChallenges vs ES_contributed (chi-square results):")
    print(chal_res_df)
    chal_res_df.to_excel(inf_writer, sheet_name="Challenges_vs_EScontrib")
else:
    print("No challenge columns or ES contribution column missing; skipping Objective 3 inferential tests.")

inf_writer.save()
print("\nInferential tables saved to:", INFERENTIAL_XLSX)

# --------------------------
# MODELLING: Logistic regression for objective 1 (Does ES contribute to early detection?)
# - outcome: ES_contributed (Yes=1 vs No=0). Remove 'Not sure' responses.
# - predictors: awareness, training, familiarity, years experience, primary role, works both, number_of_challenges, frequency, which system faster
# Save model coefficients, ORs, CIs, classification metrics, ROC plot
# --------------------------
print("\nMODELLING: Logistic regression predicting ES contribution (Objective 1)")

model_writer = pd.ExcelWriter(MODELS_XLSX, engine="openpyxl")
model_text_path = os.path.join(TEXT_DIR, "model_summary.txt")

if es_contrib_map:
    es_code_col = mapping_info["Has ES in your area contributed to early detection of poliovirus cases?"]["code_col"]
    # keep only 0/1 entries
    df_model = df[df[es_code_col].isin([0,1])]
    if len(df_model) < 30:
        print("WARNING: small sample for modelling after dropping 'Not sure' rows:", len(df_model))
    # Prepare predictor columns
    def safe_code(colkey):
        m = mapping_info.get(colkey)
        if not m:
            return None
        return m["code_col"]

    pred_code_cols = []
    for key in [
        "Are you aware of environmental surveillance (ES) activities in your area?",
        "Have you received formal training on ES?",
        "How familiar are you with the processes of ES (sewage sampling, lab testing)?",
        "Years of Experience",
        "Do you work with both AFP and ES systems?",
        "Frequency of environmental sample collection",
        "Which system has faster virus detection?",
        "Primary Role in Environmental Surveillance for Polio Eradication"
    ]:
        cc = safe_code(key)
        if cc and cc in df_model.columns:
            pred_code_cols.append(cc)
    # Add numeric count of challenges selected
    if chal_cols:
        df_model["num_challenges"] = df_model[chal_cols].sum(axis=1)
    else:
        df_model["num_challenges"] = 0

    # Build formula using categorical predictors with C()
    # Keep only predictors present
    predictors_formula = []
    for col in pred_code_cols:
        # if numeric with multiple categories (coded), treat as categorical
        # use column name without spaces for formula safety
        col_safe = re.sub(r'[^\w]', '_', col)
        df_model[col_safe] = pd.to_numeric(df_model[col], errors='coerce')
        predictors_formula.append(f"C({col_safe})")
    # num_challenges and years as numeric (if years present use its safe name)
    if "num_challenges" in df_model.columns:
        predictors_formula.append("num_challenges")
    formula = "es_outcome ~ " + " + ".join(predictors_formula) if predictors_formula else None
    # create es_outcome numeric column
    df_model["es_outcome"] = pd.to_numeric(df_model[es_code_col], errors='coerce').astype(float)
    # drop rows with missing outcome
    df_model = df_model.dropna(subset=["es_outcome"])
    if formula:
        try:
            print("Model formula:", formula)
            model = smf.logit(formula=formula.replace("es_outcome", "es_outcome"), data=df_model).fit(disp=False, maxiter=200)
            print("\nModel converged.")
            # Coeff table
            coef_table = model.summary2().tables[1]
            coef_table.to_excel(model_writer, sheet_name="logit_coef")
            # Odds ratios
            params = model.params
            conf = model.conf_int()
            or_df = pd.DataFrame({
                "coef": params,
                "OR": np.exp(params),
                "2.5%_OR": np.exp(conf[0]),
                "97.5%_OR": np.exp(conf[1]),
                "pvalue": model.pvalues
            })
            or_df.to_excel(model_writer, sheet_name="logit_oddsratios")
            print("\nLogistic regression results (coefficients):")
            print(coef_table)
            print("\nOdds ratios:")
            print(or_df)
            # Save text summary
            with open(model_text_path, "w", encoding="utf-8") as f:
                f.write(model.summary2().as_text())
            # Predictions and classification metrics
            X = model.model.exog
            probs = model.predict()
            preds = (probs >= 0.5).astype(int)
            y_true = model.model.endog.astype(int)
            class_report = classification_report(y_true, preds, zero_division=0)
            cm = confusion_matrix(y_true, preds)
            auc = roc_auc_score(y_true, probs)
            metrics_df = pd.DataFrame({
                "AUC": [auc],
                "n_obs": [len(y_true)]
            })
            metrics_df.to_excel(model_writer, sheet_name="metrics", index=False)
            # Write classification report to text file and excel
            with open(os.path.join(TEXT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(class_report)
            cr_df = pd.DataFrame([x.split() for x in class_report.splitlines() if x], dtype=object)
            # ROC plot
            fpr, tpr, _ = roc_curve(y_true, probs)
            plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, lw=2)
            plt.plot([0,1],[0,1],'k--', lw=1)
            plt.xlabel("False positive rate")
            plt.ylabel("True positive rate")
            plt.title(f"ROC curve (AUC = {auc:.3f})")
            plt.tight_layout()
            out_roc = os.path.join(PLOTS_DIR, "roc_es_contribution.png")
            plt.savefig(out_roc, dpi=300)
            plt.close()
            print("ROC saved to:", out_roc)
            # Save confusion matrix as small figure
            plt.figure(figsize=(4,4))
            sns.heatmap(cm, annot=True, fmt="d", cbar=False)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title("Confusion matrix (es_outcome)")
            plt.tight_layout()
            out_cm = os.path.join(PLOTS_DIR, "confusion_matrix_es.png")
            plt.savefig(out_cm, dpi=300)
            plt.close()
            print("Confusion matrix saved to:", out_cm)
        except Exception as e:
            print("Logistic regression failed:", e)
            with open(model_text_path, "w", encoding="utf-8") as f:
                f.write("Model failed: " + str(e))
    else:
        print("No predictors available for logistic regression.")
else:
    print("ES contribution column not available for modelling. Skipping model step.")

model_writer.save()
print("Model outputs saved to:", MODELS_XLSX)
print("Model textual summaries saved to:", model_text_path)

# --------------------------
# ADDITIONAL ANALYSES TO STRENGTHEN OBJECTIVES
# - Number of challenges per respondent (descriptive + relation to ES contribution)
# - Proportion of respondents who answered the 'correct' definition of ES (option 2)
# - Save all the key tables into a single Excel workbook for easy review
# --------------------------
print("\nADDITIONAL ANALYSES...")

extra_writer = pd.ExcelWriter(os.path.join(OUT_DIR, "extra_analysis.xlsx"), engine="openpyxl")

# 1) Number of challenges distribution
if chal_cols:
    df["num_challenges"] = df[chal_cols].sum(axis=1)
    num_chal_tab = freq_table(df["num_challenges"].fillna(0))
    print_and_save_dataframe(num_chal_tab, title="Distribution: Number of challenges selected")
    num_chal_tab.to_excel(extra_writer, sheet_name="num_challenges_dist")
    # plot histogram
    plt.figure(figsize=(7,4))
    sns.histplot(df["num_challenges"].dropna(), bins=range(0, max(4, int(df["num_challenges"].max())+2)))
    plt.xlabel("Number of challenges selected")
    plt.ylabel("Count")
    plt.title("Distribution: number of operational challenges reported")
    plt.tight_layout()
    p = os.path.join(PLOTS_DIR, "num_challenges_hist.png")
    plt.savefig(p, dpi=300)
    plt.close()
    print("Saved plot:", p)
else:
    print("No challenge columns for extra analyses.")

# 2) Correct definition of ES: proportion selecting option 2 (Monitoring and analyzing environmental hazards ...)
def_es_map = mapping_info.get("What is environmental surveillance about?")
if def_es_map:
    def_lab = def_es_map["label_col"]
    correct_label = CODEBOOK["What is environmental surveillance about?"][2]
    is_correct = df[def_lab] == correct_label
    correct_pct = is_correct.mean() * 100
    print(f"\nProportion selecting correct definition (option 2): {correct_pct:.2f}%")
    corr_tab = pd.DataFrame({"Correct_definition_percent": [round(correct_pct,2)], "n": [len(df)]})
    corr_tab.to_excel(extra_writer, sheet_name="Correct_definition_pct")
    # Save distribution
    freq_table(df[def_lab].fillna("Missing")).to_excel(extra_writer, sheet_name="ES_def_distribution")
else:
    print("Definition-of-ES variable not found.")

extra_writer.save()
print("Extra analysis saved to:", os.path.join(OUT_DIR, "extra_analysis.xlsx"))

# --------------------------
# INTERPRETATION & RECOMMENDATIONS (text file)
# --------------------------
recommend_path = os.path.join(TEXT_DIR, "interpretation_recommendations.txt")
with open(recommend_path, "w", encoding="utf-8") as f:
    f.write("Interpretation Guide & Recommendations\n")
    f.write("="*80 + "\n\n")
    f.write("This file summarizes how to read the outputs and recommended actions based on findings.\n\n")
    f.write("1) Objective 1 — Contribution of ES to early detection\n")
    f.write("   - Descriptive tables in 'descriptive_tables.xlsx' show percent reporting ES contributed.\n")
    f.write("   - Chi-square tests (in 'inferential_tables.xlsx') display associations between ES contribution and awareness, training and familiarity.\n")
    f.write("   - Logistic regression (model_results.xlsx) provides adjusted odds ratios for predictors of ES contribution; AUC shows discrimination.\n\n")
    f.write("2) Objective 2 — Comparison with AFP\n")
    f.write("   - State-by-state tables show percent reporting ES 'More effective' vs 'Equally/Less/Don't know'.\n")
    f.write("   - For selected states (top-response states), see 'Effectiveness_by_state' in inferential_tables.xlsx.\n\n")
    f.write("3) Objective 3 — Operational Challenges\n")
    f.write("   - Challenges summary and per-challenge chi-square results are in inferential_tables.xlsx.\n")
    f.write("   - If a challenge is strongly associated (small p-value) with lack of ES contribution, prioritize that operational area.\n\n")
    f.write("General recommendations (template):\n")
    f.write(" - Strengthen training in states with lower training rates and lower awareness (see state_awareness_pct.png).\n")
    f.write(" - Target logistical/transport and lab delays (commonly selected challenges) with resource allocation and SOP improvements.\n")
    f.write(" - Consider targeted community awareness campaigns in states with low ES awareness.\n\n")
    f.write("See the plots in outputs/plots for visual summaries and outputs/*.xlsx for tables.\n")

print("\nInterpretation & recommendations saved to:", recommend_path)

print("\nALL TASKS COMPLETE. Key output locations:")
print(" - Descriptive tables (Excel):", DESCRIPTIVE_XLSX)
print(" - Inferential tables (Excel):", INFERENTIAL_XLSX)
print(" - Model outputs (Excel):", MODELS_XLSX)
print(" - Extra analyses (Excel):", os.path.join(OUT_DIR, "extra_analysis.xlsx"))
print(" - Plots:", PLOTS_DIR)
print(" - Text summaries & recommendations:", TEXT_DIR)

print("\nIf you want any of the following additions, tell me which and I'll update the script:")
print(" - Export all figures as vector (SVG/PDF) for publication.")
print(" - Multilevel (mixed-effects) logistic regression with random intercept for state.")
print(" - Random forest classifier for robustness & variable importance.")
print(" - A single combined Word/PDF report with figures and tables formatted for publication.")
