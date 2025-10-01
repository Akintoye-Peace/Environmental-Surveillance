import os
import re
import sys
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
CSV_PATHS_TO_TRY = [
    r"C:\Users\everp\Documents\Surveillance\surveillance_recoded.csv",
    r"C:\Users\everp\Documents\Surveillance\surveilance_recoded.csv",
    r"./surveillance_recoded.csv",
    r"./surveilance_recoded.csv"
]
OUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
TEXT_DIR = os.path.join(OUT_DIR, "text")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

DESCRIPTIVE_XLSX = os.path.join(TABLES_DIR, "descriptive_tables.xlsx")
INFERENTIAL_XLSX = os.path.join(TABLES_DIR, "inferential_tables.xlsx")
MODELS_XLSX = os.path.join(TABLES_DIR, "model_results.xlsx")
EXTRA_XLSX = os.path.join(TABLES_DIR, "extra_analysis.xlsx")

# --------------------------
# CODEBOOK (numeric -> label)
# --------------------------
CODEBOOK = {
    "Gender": {1: "Male", 2: "Female"},
    "Age Group": {1: "18-30", 2: "31-40", 3: "41-50", 4: "51+"},
    "Current Position": {
        1: "Surveillance Officer",
        2: "Laboratory Scientist",
        3: "Environmental Health Officer",
        4: "Health Facility Worker",
        5: "Other"
    },
    "Years of Experience": {
        1: "Less than 5 years",
        2: "5-10 years",
        3: "11-15 years",
        4: "Over 15 years"
    },
    "Are you aware of environmental surveillance (ES) activities in your area?": {1: "Yes", 0: "No"},
    "How familiar are you with the processes of ES (sewage sampling, lab testing)?": {
        3: "Very familiar", 2: "Somewhat familiar", 1: "Not familiar"
    },
    "What is environmental surveillance about?": {
        1: "Tracking wildlife populations",
        2: "Monitoring and analyzing environmental hazards",
        3: "Enforcing workplace safety regulations",
        4: "Studying genetic diseases"
    },
    "Have you received formal training on ES?": {1: "Yes", 0: "No"},
    "How effective is ES in detecting poliovirus compared to  Acute Flaccid Paralysis (AFP) ?": {
        3: "More effective", 2: "Equally effective", 1: "Less effective", 0: "Don't know"
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
# HELPERS
# --------------------------
_punct_re = re.compile(r"[^\w\s]")

def normalize_text(s):
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = _punct_re.sub(" ", s)
    s = re.sub(r"\s+", " ", s)
    return s

def find_column(df, patterns):
    if isinstance(patterns, str):
        patterns = [patterns]
    col_norm = {c: normalize_text(c) for c in df.columns}
    for pat in patterns:
        pn = normalize_text(pat)
        for col, cn in col_norm.items():
            if pn in cn:
                return col
    for pat in patterns:
        words = [w for w in normalize_text(pat).split() if len(w) > 2]
        for col, cn in col_norm.items():
            if all(w in cn for w in words):
                return col
    return None

def safe_name(s):
    return re.sub(r"[^\w\-]", "_", str(s))[:80]

def save_png(fig, fname):
    path = os.path.join(PLOTS_DIR, safe_name(fname) + ".png")
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path

def summarize_categorical(series):
    counts = series.value_counts(dropna=False)
    perc = (counts / counts.sum() * 100).round(2)
    return pd.DataFrame({"Count": counts, "Percent": perc})

def plot_categorical(series, title, fname):
    counts = series.value_counts(dropna=False)
    labels = counts.index.astype(str)
    values = counts.values

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=np.arange(len(values)), y=values, ax=ax, palette="viridis")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha='right')

    ax.set_ylabel("Count")
    ax.set_xlabel("")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, safe_name(fname) + ".png"), dpi=300)
    plt.close()


def write_excel_sheets(writer, dict_of_dfs, prefix=""):
    """Write multiple DataFrames to Excel with unique sheet names."""
    for i, (sheet, df_) in enumerate(dict_of_dfs.items(), start=1):
        base = f"{prefix}{safe_name(sheet)}"
        sheet_name = f"{base}_{i}" if base in writer.book.sheetnames else base
        try:
            df_.to_excel(writer, sheet_name=sheet_name[:31])
        except Exception as e:
            pd.DataFrame({sheet: [str(e)]}).to_excel(
                writer, sheet_name=(sheet_name + "_err")[:31]
            )

# LOAD CSV

csv_found = None
for p in CSV_PATHS_TO_TRY:
    if os.path.exists(p):
        csv_found = p
        break
if csv_found is None:
    print("ERROR: Could not find your CSV file. Tried these paths:")
    for p in CSV_PATHS_TO_TRY:
        print("  -", p)
    print("Please move your CSV to one of these paths or update CSV_PATHS_TO_TRY in the script.")
    sys.exit(1)

print("Loading data from:", csv_found)
df_raw = pd.read_csv(csv_found, dtype=str)
df_raw.columns = [c.strip() for c in df_raw.columns]
df_raw = df_raw.applymap(lambda x: x.strip() if isinstance(x, str) else x)


# MAP codebook 

df = df_raw.copy()
mapping_info = {}

for cb_col, cmap in CODEBOOK.items():
    found = find_column(df, cb_col)
    if not found:
        found = find_column(df, " ".join(cb_col.split()[:4]))
    if not found:
        print(f"NOTICE: column matching '{cb_col}' not found. Skipping mapping for this variable.")
        continue
    label_col = found + "_label"
    code_col = found + "_code"
    mapping_info[cb_col] = {"dataset_column": found, "label_col": label_col, "code_col": code_col}
    inv = {v.lower(): k for k, v in cmap.items()}
    labs = []
    codes = []
    for v in df[found].fillna("").astype(str).values:
        s = v.strip()
        if s == "" or s.lower() in ["nan", "none"]:
            labs.append(np.nan); codes.append(np.nan); continue
        
        if re.fullmatch(r"-?\d+", s):
            n = int(s)
            if n in cmap:
                labs.append(cmap[n]); codes.append(n)
            else:
                labs.append(s); codes.append(np.nan)
            continue
        sl = s.lower()
        if sl in inv:
            code = inv[sl]; labs.append(cmap[code]); codes.append(code); continue
        matched = False
        for k_label in cmap.values():
            if normalize_text(k_label) == normalize_text(s):
                labs.append(k_label); codes.append([k for k,v in cmap.items() if v==k_label][0])
                matched = True; break
        if matched: continue
        digits = re.findall(r"\d+", s)
        if digits:
            n = int(digits[0])
            if n in cmap:
                labs.append(cmap[n]); codes.append(n); continue
        labs.append(s); codes.append(np.nan)
    df[label_col] = labs
    df[code_col] = codes


# State/location column

loc_col = find_column(df, ["Location/State of Primary Assignment", "Location", "State of Primary Assignment"])
if loc_col:
    df["state_label"] = df[loc_col].replace({"": np.nan})
else:
    df["state_label"] = np.nan
    print("NOTICE: no state/location column found; state analyses will be skipped.")


# Challenges parsing (binary)

challenge_bin_cols = [c for c in df.columns if normalize_text(c).startswith("challenge_")]
if challenge_bin_cols:
    for c in challenge_bin_cols:
        df[c] = pd.to_numeric(df[c].replace({"":0, "nan":0}), errors='coerce').fillna(0).astype(int)
else:
    chal_col = find_column(df, "Challenges encountered in ES")
    if chal_col:
        for opt in CHALLENGE_OPTIONS:
            cname = "Challenge_" + re.sub(r"[^\w]", "_", opt).strip("_")
            df[cname] = 0
        for idx, val in df[chal_col].fillna("").astype(str).items():
            if val.strip()=="":
                continue
            parts = [p.strip().lower() for p in re.split(r"[;,/|]", val) if p.strip()!=""]
            for opt in CHALLENGE_OPTIONS:
                if any(opt.lower() in p for p in parts):
                    cname = "Challenge_" + re.sub(r"[^\w]", "_", opt).strip("_")
                    df.at[idx, cname] = 1
        challenge_bin_cols = [c for c in df.columns if normalize_text(c).startswith("challenge_")]
    else:
        challenge_bin_cols = []


# Utility

def freq_table(series, name=None, dropna=False):
    name = name or series.name
    vc = series.value_counts(dropna=dropna)
    total = vc.sum()
    pct = (vc / total * 100).round(2)
    table = pd.DataFrame({"Count": vc, "Percent": pct})
    return table

def print_table(df_tab, title=None):
    if title:
        print("\n" + "="*100)
        print(title)
        print(df_tab.to_string())
        print("="*100 + "\n")
    else:
        print(df_tab.to_string())


# DESCRIPTIVE: Sociodemographics 

print("\n==== SOCIODEMOGRAPHIC SUMMARY ====\n")
sociodemo_vars = ["Gender", "Age Group", "Current Position", "Years of Experience", "Location/State of Primary Assignment"]
sociodemo_tables = {}
for var in sociodemo_vars:
    
    mapping = mapping_info.get(var)
    if mapping:
        tab = freq_table(df[mapping["label_col"]].fillna("Missing"))
        print_table(tab, title=f"Sociodemographic: {var} (from '{mapping['dataset_column']}')")
        sociodemo_tables[var] = tab
    else:
        found = find_column(df_raw, var)
        if found:
            tab = freq_table(df_raw[found].fillna("Missing"))
            print_table(tab, title=f"Sociodemographic (raw): {found}")
            sociodemo_tables[var] = tab
        else:
            print(f" - {var}: column not found.")

# Save sociodemographics to Excel
with pd.ExcelWriter(DESCRIPTIVE_XLSX, engine="openpyxl", mode="w") as w:
    write_excel_sheets(w, sociodemo_tables, prefix="Socio_")

# Also save a CSV copy of cleaned labelled data
cleaned_csv = os.path.join(OUT_DIR, "cleaned_labelled_data.csv")
df.to_csv(cleaned_csv, index=False)


# DESCRIPTIVE: All codebook variables

print("\n==== FULL DESCRIPTIVE TABLES (codebook variables) ====\n")
desc_tables = {}
for cb in CODEBOOK.keys():
    mapping = mapping_info.get(cb)
    if mapping:
        lab = mapping["label_col"]
        tab = freq_table(df[lab].fillna("Missing"))
        print_table(tab, title=f"Variable: {cb} (from '{mapping['dataset_column']}')")
        desc_tables[cb] = tab
        # plot bar with percent labels
        counts = tab["Count"]
        labels = counts.index.astype(str)
        fig, ax = plt.subplots(figsize=(8,5))
        sns.barplot(x=labels, y=counts.values, ax=ax)
        ax.set_title(cb)
        ax.set_ylabel("Count")
        ax.set_xticklabels(labels, rotation=35, ha='right')
        # annotate percent on bars
        for i, v in enumerate(counts.values):
            pct = tab["Percent"].iloc[i]
            ax.text(i, v + max(counts.values)*0.01, f"{int(v)}\n({pct}%)", ha='center', va='bottom', fontsize=8)
        save_png(fig, f"desc_{cb}")
    else:
        found = find_column(df_raw, cb)
        if found:
            tab = freq_table(df_raw[found].fillna("Missing"))
            print_table(tab, title=f"Variable (raw): {found}")
            desc_tables[found] = tab
            # plot
            counts = tab["Count"]
            labels = counts.index.astype(str)
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x=labels, y=counts.values, ax=ax)
            ax.set_title(found)
            ax.set_ylabel("Count")
            ax.set_xticklabels(labels, rotation=35, ha='right')
            for i, v in enumerate(counts.values):
                pct = tab["Percent"].iloc[i]
                ax.text(i, v + max(counts.values)*0.01, f"{int(v)}\n({pct}%)", ha='center', va='bottom', fontsize=8)
            save_png(fig, f"desc_{found}")

# Append all descriptive tables to descriptive workbook
with pd.ExcelWriter(DESCRIPTIVE_XLSX, engine="openpyxl", mode="a") as w:
    write_excel_sheets(w, desc_tables, prefix="Desc_")


# CHALLENGES

if challenge_bin_cols:
    chal_counts = df[challenge_bin_cols].sum().sort_values(ascending=False)
    chal_df = pd.DataFrame({"Count": chal_counts, "Percent": (chal_counts / len(df) * 100).round(2)})
    print_table(chal_df, title="Operational challenges (binary counts & percent)")
    # save to descriptive xlsx
    with pd.ExcelWriter(DESCRIPTIVE_XLSX, engine="openpyxl", mode="a") as w:
        chal_df.to_excel(w, sheet_name="Challenges_Summary")
    # plot horizontal bar with counts
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=chal_counts.values, y=[c.replace("Challenge_","") for c in chal_counts.index], ax=ax)
    ax.set_xlabel("Number of respondents selecting challenge")
    ax.set_title("Challenges encountered in ES (totals)")
    for i, v in enumerate(chal_counts.values):
        ax.text(v + max(chal_counts.values)*0.01, i, str(int(v)), va='center')
    save_png(fig, "challenges_counts")
else:
    print("No challenge columns found to summarize.")


# STATE-BY-STATE SUMMARY (Awareness, Training, ES contribution, Effectiveness)

print("\n==== STATE-BY-STATE SUMMARIES ====\n")
state_tables = {}
if df["state_label"].notna().sum() > 0:
    
    metric_keys = {
        "Awareness": "Are you aware of environmental surveillance (ES) activities in your area?",
        "Training": "Have you received formal training on ES?",
        "ES_contributed": "Has ES in your area contributed to early detection of poliovirus cases?",
        "Effectiveness": "How effective is ES in detecting poliovirus compared to  Acute Flaccid Paralysis (AFP) ?"
    }
    for mname, varname in metric_keys.items():
        mapping = mapping_info.get(varname)
        if mapping:
            labcol = mapping["label_col"]
            per_state = pd.crosstab(df["state_label"], df[labcol], normalize='index').fillna(0)
            per_state_pct = (per_state * 100).round(2)
            print(f"State-by-state percent distribution for {mname}:")
            print(per_state_pct)
            state_tables[f"{mname}_by_state"] = per_state_pct
            # plot top states by count
            # create percent aware plot for 'Yes' where applicable
            if "Yes" in per_state.columns:
                yes_pct = per_state["Yes"].sort_values(ascending=False)
                fig, ax = plt.subplots(figsize=(10,6))
                sns.barplot(x=(yes_pct.values*100), y=yes_pct.index, ax=ax)
                ax.set_xlabel("Percent Yes (%)")
                ax.set_title(f"Percent '{mname}=Yes' by state")
                save_png(fig, f"{mname}_by_state_pct")
    with pd.ExcelWriter(DESCRIPTIVE_XLSX, engine="openpyxl", mode="a") as w:
        write_excel_sheets(w, state_tables)
else:
    print("No state labels available; skipping state-by-state summaries.")


# INFERENTIAL ANALYSES (chi-square / fisher for 2x2) 

print("\n==== INFERENTIAL ANALYSES ====\n")
inf_results = {}
inf_ctables = {}

# helper: sanitize and ensure unique sheet names for the writer
def sanitize_sheet_name(name):
    # Excel invalid characters: : \ / ? * [ ]
    # Also trim length to 31
    s = re.sub(r"[:\\\/\?\*\[\]]", "_", str(name))
    s = re.sub(r"[\n\r\t]", " ", s)
    s = s.strip()[:31]
    if s == "":
        s = "sheet"
    return s

def get_unique_sheet_name(writer, base):
    base = sanitize_sheet_name(base)[:31]
    existing = set()
    try:
        existing = set(writer.book.sheetnames)
    except Exception:
        # if writer has no .book (older pandas), keep simple
        existing = set()
    if base not in existing:
        return base
    # append numeric suffix to avoid collision
    for i in range(1, 100):
        candidate = f"{base[:26]}_{i}"[:31]
        if candidate not in existing:
            return candidate
    # fallback
    return base[:31]

def test_association(colA, colB, df_local):
    a = df_local[colA].fillna("Missing")
    b = df_local[colB].fillna("Missing")
    ct = pd.crosstab(a, b)
    # compute expected counts for diagnostics if possible
    try:
        if ct.size == 0:
            return ct, {"error": "empty table"}, None
        if ct.shape == (2,2):
            # Fisher exact for 2x2
            table_vals = ct.values
            odds, p = stats.fisher_exact(table_vals)
            # expected for 2x2:
            chi2, chi_p, dof, expected = stats.chi2_contingency(ct)
            expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
            # check expected counts <5
            low_expected = (expected_df < 5).sum().sum()
            info = {"test":"fisher_exact", "p":float(p), "oddsratio":float(odds),
                    "chi2_expected_low_cells": int(low_expected)}
            return ct, info, expected_df
        else:
            chi2, p, dof, expected = stats.chi2_contingency(ct)
            expected_df = pd.DataFrame(expected, index=ct.index, columns=ct.columns)
            low_expected = (expected_df < 5).sum().sum()
            info = {"test":"chi2", "chi2": float(chi2), "p": float(p), "dof": int(dof),
                    "chi2_expected_low_cells": int(low_expected)}
            return ct, info, expected_df
    except Exception as e:
        return ct, {"error": str(e)}, None

# open writer once and use unique sheet naming
inf_writer = pd.ExcelWriter(INFERENTIAL_XLSX, engine="openpyxl", mode="w")

es_map = mapping_info.get("Has ES in your area contributed to early detection of poliovirus cases?")
assoc_vars = [
    "Are you aware of environmental surveillance (ES) activities in your area?",
    "Have you received formal training on ES?",
    "How familiar are you with the processes of ES (sewage sampling, lab testing)?",
    "Primary Role in Environmental Surveillance for Polio Eradication",
    "Do you work with both AFP and ES systems?"
]
if es_map:
    es_lab = es_map["label_col"]
    for v in assoc_vars:
        m = mapping_info.get(v)
        if not m:
            print(f"Assoc var '{v}' not found; skipping.")
            continue
        ct, res, expected_df = test_association(m["label_col"], es_lab, df)
        base_name = f"{v}_vs_EScontrib"
        sheet_ct = get_unique_sheet_name(inf_writer, base_name)
        # Write contingency table
        try:
            ct.to_excel(inf_writer, sheet_name=sheet_ct)
        except Exception as e:
            # fallback: write a note
            pd.DataFrame({f"error_{sheet_ct}":[str(e)]}).to_excel(inf_writer, sheet_name=get_unique_sheet_name(inf_writer, sheet_ct + "_err"))
        # write expected counts (if available)
        if expected_df is not None:
            try:
                expected_sheet = get_unique_sheet_name(inf_writer, sheet_ct + "_expected")
                expected_df.to_excel(inf_writer, sheet_name=expected_sheet)
            except Exception:
                pass
        # write results summary
        try:
            resdf = pd.DataFrame([res])
            res_sheet = get_unique_sheet_name(inf_writer, sheet_ct + "_res")
            resdf.to_excel(inf_writer, sheet_name=res_sheet, index=False)
        except Exception:
            pass
        inf_ctables[base_name] = ct
        inf_results[base_name] = res
        # print with more readable formatting
        print(f"Association: {v} vs ES contribution -> {res}")
        if expected_df is not None and res.get("chi2_expected_low_cells", 0) > 0:
            print(f"  NOTE: {res['chi2_expected_low_cells']} expected cell(s) < 5 — chi-square assumptions may be violated.")
else:
    print("ES contribution variable not found; skipping Objective 1 inferential tests.")

# Objective 2: Effectiveness vs 'which is faster' and by state (chi-square)
eff_map = mapping_info.get("How effective is ES in detecting poliovirus compared to  Acute Flaccid Paralysis (AFP) ?")
which_map = mapping_info.get("Which system has faster virus detection?")
if eff_map and which_map:
    ct, res, expected_df = test_association(eff_map["label_col"], which_map["label_col"], df)
    sheet_base = "Effectiveness_vs_WhichFaster"
    sheet_ct = get_unique_sheet_name(inf_writer, sheet_base)
    try:
        ct.to_excel(inf_writer, sheet_name=sheet_ct)
    except Exception as e:
        pd.DataFrame({f"error_{sheet_ct}":[str(e)]}).to_excel(inf_writer, sheet_name=get_unique_sheet_name(inf_writer, sheet_ct + "_err"))
    if expected_df is not None:
        try:
            expected_df.to_excel(inf_writer, sheet_name=get_unique_sheet_name(inf_writer, sheet_ct + "_expected"))
        except Exception:
            pass
    pd.DataFrame([res]).to_excel(inf_writer, sheet_name=get_unique_sheet_name(inf_writer, sheet_ct + "_res"), index=False)
    print("Effectiveness vs WhichFaster:", res)
    if expected_df is not None and res.get("chi2_expected_low_cells", 0) > 0:
        print(f"  NOTE: {res['chi2_expected_low_cells']} expected cell(s) < 5 — chi-square assumptions may be violated.")
else:
    print("Effectiveness or WhichFaster variable missing; skipping that test.")

# Objective 3: Challenges vs ES contribution
if challenge_bin_cols and es_map:
    chal_results = []
    for c in challenge_bin_cols:
        ct, res, expected_df = test_association(c, es_lab, df)
        base_name = f"{c}_vs_EScontrib"
        sheet_ct = get_unique_sheet_name(inf_writer, base_name)
        try:
            ct.to_excel(inf_writer, sheet_name=sheet_ct)
        except Exception as e:
            pd.DataFrame({f"error_{sheet_ct}":[str(e)]}).to_excel(inf_writer, sheet_name=get_unique_sheet_name(inf_writer, sheet_ct + "_err"))
        if expected_df is not None:
            try:
                expected_df.to_excel(inf_writer, sheet_name=get_unique_sheet_name(inf_writer, sheet_ct + "_expected"))
            except Exception:
                pass
        pd.DataFrame([res]).to_excel(inf_writer, sheet_name=get_unique_sheet_name(inf_writer, sheet_ct + "_res"), index=False)
        chal_results.append({"challenge": c, **res})
        if expected_df is not None and res.get("chi2_expected_low_cells", 0) > 0:
            print(f"Challenge {c}: NOTE {res['chi2_expected_low_cells']} expected cell(s) < 5 — consider collapsing categories or using exact tests.")
    chal_res_df = pd.DataFrame(chal_results)
    chal_res_df.to_excel(inf_writer, sheet_name=get_unique_sheet_name(inf_writer, "Challenges_vs_EScontrib"), index=False)
    print("Challenge associations saved.")
else:
    print("Skipping challenge vs ES contribution tests (missing data).")

inf_writer.close()
print("Inferential tables saved to:", INFERENTIAL_XLSX)



# MODELLING: Logistic regression for Objective 1 (ES contribution)

print("\n==== MODELLING: Logistic regression for ES contribution ====\n")
model_writer = pd.ExcelWriter(MODELS_XLSX, engine="openpyxl", mode="w")
model_textfile = os.path.join(TEXT_DIR, "model_summary.txt")

if es_map:
    es_code_col = mapping_info["Has ES in your area contributed to early detection of poliovirus cases?"]["code_col"]
    # keep only 0/1
    df_model = df[df[es_code_col].isin([0,1])].copy()
    if df_model.shape[0] < 30:
        print("WARNING: small sample for modelling:", df_model.shape[0])
    # prepare predictors (code columns)
    pred_keys = [
        "Are you aware of environmental surveillance (ES) activities in your area?",
        "Have you received formal training on ES?",
        "How familiar are you with the processes of ES (sewage sampling, lab testing)?",
        "Years of Experience",
        "Do you work with both AFP and ES systems?",
        "Frequency of environmental sample collection",
        "Which system has faster virus detection?",
        "Primary Role in Environmental Surveillance for Polio Eradication"
    ]
    pred_code_cols = []
    for pk in pred_keys:
        m = mapping_info.get(pk)
        if m and m["code_col"] in df_model.columns:
            pred_code_cols.append(m["code_col"])
    # add numeric number of challenges
    if challenge_bin_cols:
        df_model["num_challenges"] = df_model[challenge_bin_cols].sum(axis=1)
    else:
        df_model["num_challenges"] = 0
    # Build formula
    formula_terms = []
    for col in pred_code_cols:
        safecol = re.sub(r"[^\w]", "_", col)
        df_model[safecol] = pd.to_numeric(df_model[col], errors='coerce')
        formula_terms.append(f"C({safecol})")
    formula_terms.append("num_challenges")
    if len(formula_terms) == 0:
        print("No predictors found for modelling.")
    else:
        formula = "es_outcome ~ " + " + ".join(formula_terms)
        df_model["es_outcome"] = pd.to_numeric(df_model[es_code_col], errors='coerce')
        df_model = df_model.dropna(subset=["es_outcome"])
        try:
            model = smf.logit(formula=formula, data=df_model).fit(disp=False, maxiter=200)
            llr = model.llr
            llr_p = model.llr_pvalue     # p-value
            df_model = int(model.df_model)  # df
            pseudo_r2 = model.prsquared

            fit_stats = pd.DataFrame({
                "Chi2":[llr],
                "df":[df_model],
                "p_value":[llr_p],
                "Pseudo_R2":[pseudo_r2]
            })
            fit_stats.to_excel(model_writer, sheet_name="fit_stats", index=False)
            coef = model.summary2().tables[1]
            coef.to_excel(model_writer, sheet_name="logit_coef")
            params = model.params
            conf = model.conf_int()
            or_df = pd.DataFrame({
                "coef": params,
                "OR": np.exp(params),
                "2.5%_OR": np.exp(conf[0]),
                "97.5%_OR": np.exp(conf[1]),
                "pvalue": model.pvalues
            })
            or_df.to_excel(model_writer, sheet_name="logit_or")
            # predictions & metrics
            probs = model.predict()
            preds = (probs >= 0.5).astype(int)
            y = model.model.endog.astype(int)
            classrep = classification_report(y, preds, zero_division=0)
            with open(os.path.join(TEXT_DIR, "classification_report.txt"), "w", encoding="utf-8") as f:
                f.write(classrep)
            metrics_df = pd.DataFrame({"AUC":[roc_auc_score(y, probs)], "n":[len(y)]})
            metrics_df.to_excel(model_writer, sheet_name="metrics", index=False)
            # save text summary
            with open(model_textfile, "w", encoding="utf-8") as f:
                f.write(model.summary2().as_text())
            # ROC plot
            fpr, tpr, _ = roc_curve(y, probs)
            fig, ax = plt.subplots(figsize=(6,6))
            ax.plot(fpr, tpr, lw=2, label=f"AUC={roc_auc_score(y, probs):.3f}")
            ax.plot([0,1],[0,1],'k--', lw=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC: ES contribution model")
            ax.legend(loc="lower right")
            save_png(fig, "roc_es_contribution")
            # confusion matrix heatmap
            cm = confusion_matrix(y, preds)
            fig, ax = plt.subplots(figsize=(4,4))
            sns.heatmap(cm, annot=True, fmt="d", cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion matrix")
            save_png(fig, "confusion_matrix_es")
            print("Model fitted and outputs saved.")
        except Exception as e:
            print("Model failed:", e)
            with open(model_textfile, "w", encoding="utf-8") as f:
                f.write("Model failed: " + str(e))
else:
    print("ES contribution map missing; cannot fit model.")

model_writer.close()
print("Model outputs saved to:", MODELS_XLSX)


# EXTRA ANALYSIS: Cross-tabs & Challenge Models

print("\n==== EXTRA ANALYSIS: Cross-tabs and Challenge Models ====\n")

extra_writer = pd.ExcelWriter(EXTRA_XLSX, engine="openpyxl", mode="w")


# Cross-tab: ES vs AFP perceived effectiveness

eff_map = mapping_info.get("How effective is ES in detecting poliovirus compared to  Acute Flaccid Paralysis (AFP) ?")
which_map = mapping_info.get("Which system has faster virus detection?")
if eff_map and which_map:
    eff_lab = eff_map["label_col"]
    which_lab = which_map["label_col"]
    cross_tab = pd.crosstab(df[eff_lab], df[which_lab], normalize="index") * 100
    cross_tab = cross_tab.round(2)
    cross_tab.to_excel(extra_writer, sheet_name="Effectiveness_vs_AFP")

    print("Cross-tab: ES effectiveness vs AFP faster detection")
    print(cross_tab)

    # Bar chart (stacked)
    cross_tab_plot = pd.crosstab(df[eff_lab], df[which_lab])
    cross_tab_plot.plot(kind="bar", stacked=True, figsize=(10,6), colormap="viridis")
    plt.ylabel("Count")
    plt.title("ES effectiveness vs AFP faster detection")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    save_png(plt.gcf(), "CrossTab_Effectiveness_vs_AFP")
    plt.close()


# Logistic regression: Challenges predicting non-participation

if challenge_bin_cols and es_map:
    es_code_col = es_map["code_col"]

    # Binary outcome: ES contribution (0/1 only)
    df_chal_model = df[df[es_code_col].isin([0,1])].copy()
    df_chal_model["es_outcome"] = pd.to_numeric(df_chal_model[es_code_col], errors="coerce")

    # Predictor: individual challenges
    formula_terms = []
    for c in challenge_bin_cols:
        safecol = re.sub(r"[^\w]", "_", c)
        df_chal_model[safecol] = pd.to_numeric(df_chal_model[c], errors="coerce").fillna(0).astype(int)
        formula_terms.append(safecol)

    if len(formula_terms) > 0 and df_chal_model.shape[0] > 20:
        formula = "es_outcome ~ " + " + ".join(formula_terms)
        try:
            chal_model = smf.logit(formula=formula, data=df_chal_model).fit(disp=False, maxiter=200)
            print(chal_model.summary2())
            # Save outputs
            coef = chal_model.summary2().tables[1]
            coef.to_excel(extra_writer, sheet_name="Challenge_Logit_Coef")
            params = chal_model.params
            conf = chal_model.conf_int()
            or_df = pd.DataFrame({
                "coef": params,
                "OR": np.exp(params),
                "2.5%_OR": np.exp(conf[0]),
                "97.5%_OR": np.exp(conf[1]),
                "pvalue": chal_model.pvalues
            })
            or_df.to_excel(extra_writer, sheet_name="Challenge_Logit_OR")
        except Exception as e:
            print("Challenge logistic regression failed:", e)

extra_writer.close()
print("Extra analysis saved to:", EXTRA_XLSX)



# STATE-BY-STATE CROSS-TABS AND VISUALS

print("\n==== STATE-BY-STATE CROSS-TABS AND VISUALS ====\n")

state_writer = pd.ExcelWriter(EXTRA_XLSX, engine="openpyxl", mode="a")

if df["state_label"].notna().sum() > 0:
    states = df["state_label"].dropna().unique()
    
    # Objective 1: Contribution of ES to early detection
    es_map = mapping_info.get("Has ES in your area contributed to early detection of poliovirus cases?")
    if es_map:
        es_lab = es_map["label_col"]
        state_es = pd.crosstab(df["state_label"], df[es_lab], normalize="index")*100
        state_es = state_es.round(2)
        state_es.to_excel(state_writer, sheet_name="ES_Contribution_by_State")
        print("Objective 1: ES contribution by state (%)")
        print(state_es)
        
        # Bar chart: Percent Yes per state
        if "Yes" in state_es.columns:
            fig, ax = plt.subplots(figsize=(10,6))
            sns.barplot(x=state_es.index, y=state_es["Yes"].values, palette="viridis", ax=ax)
            ax.set_ylabel("Percent Yes (%)")
            ax.set_xlabel("State")
            ax.set_title("Objective 1: ES contribution to early detection by State")
            plt.xticks(rotation=35, ha='right')
            save_png(fig, "Statewise_ES_Contribution")
            plt.close()

    # Objective 2: Compare ES vs AFP effectiveness per state
    eff_map = mapping_info.get("How effective is ES in detecting poliovirus compared to  Acute Flaccid Paralysis (AFP) ?")
    which_map = mapping_info.get("Which system has faster virus detection?")
    if eff_map and which_map:
        eff_lab = eff_map["label_col"]
        which_lab = which_map["label_col"]
        for state in states:
            df_s = df[df["state_label"]==state]
            ct = pd.crosstab(df_s[eff_lab], df_s[which_lab], normalize="index")*100
            ct = ct.round(2)
            ct_name = f"Effectiveness_{state}"
            ct.to_excel(state_writer, sheet_name=ct_name[:31])
            
            # Stacked bar chart
            ct_plot = pd.crosstab(df_s[eff_lab], df_s[which_lab])
            ct_plot.plot(kind="bar", stacked=True, figsize=(8,5), colormap="viridis")
            plt.title(f"Objective 2: ES vs AFP effectiveness - {state}")
            plt.ylabel("Count")
            plt.xticks(rotation=35, ha="right")
            save_png(plt.gcf(), f"ES_vs_AFP_{state}")
            plt.close()
    
    # Objective 3: Operational challenges by state
    challenge_bin_cols = [c for c in df.columns if normalize_text(c).startswith("challenge_")]
    if challenge_bin_cols:
        for state in states:
            df_s = df[df["state_label"]==state]
            chal_counts = df_s[challenge_bin_cols].sum().sort_values(ascending=False)
            chal_df = pd.DataFrame({"Count": chal_counts, 
                                    "Percent": (chal_counts / len(df_s)*100).round(2)})
            chal_df.to_excel(state_writer, sheet_name=f"Challenges_{state}"[:31])
            
            # Horizontal bar chart
            fig, ax = plt.subplots(figsize=(8,5))
            sns.barplot(x=chal_df["Count"], y=[c.replace("Challenge_","") for c in chal_df.index], ax=ax)
            ax.set_xlabel("Number of respondents selecting challenge")
            ax.set_title(f"Objective 3: Operational challenges - {state}")
            for i, v in enumerate(chal_df["Count"]):
                ax.text(v + 0.5, i, str(int(v)), va='center')
            save_png(fig, f"Challenges_{state}")
            plt.close()
else:
    print("No state labels available; skipping state-wise analyses.")

state_writer.close()
print("State-wise cross-tabs and visualizations saved to:", EXTRA_XLSX)


# HIGH-RISK STATE IDENTIFICATION

print("\n==== HIGH-RISK STATE IDENTIFICATION ====\n")

risk_writer = pd.ExcelWriter(EXTRA_XLSX, engine="openpyxl", mode="a")
high_risk_summary = []

if df["state_label"].notna().sum() > 0:

    states = df["state_label"].dropna().unique()
    
    # Overall mean challenges for reference
    if challenge_bin_cols:
        overall_chal_mean = df[challenge_bin_cols].sum(axis=1).mean()
    else:
        overall_chal_mean = 0

    es_lab = mapping_info.get("Has ES in your area contributed to early detection of poliovirus cases?")["label_col"]
    which_lab = mapping_info.get("Which system has faster virus detection?")["label_col"]

    for state in states:
        df_s = df[df["state_label"]==state]
        
        # Objective 1: ES contribution (% Yes)
        if es_lab in df_s.columns:
            yes_pct = (df_s[es_lab]=="Yes").mean()*100
        else:
            yes_pct = np.nan

        # Objective 2: Virus detection speed (AFP faster %)
        if which_lab in df_s.columns:
            if "Acute Flaccid Paralysis (AFP)" in df_s[which_lab].unique():
                afp_faster_pct = (df_s[which_lab]=="Acute Flaccid Paralysis (AFP)").mean()*100
            else:
                afp_faster_pct = np.nan
        else:
            afp_faster_pct = np.nan

        # Objective 3: Average number of challenges
        if challenge_bin_cols:
            avg_challenges = df_s[challenge_bin_cols].sum(axis=1).mean()
        else:
            avg_challenges = 0

        # Determine if high-risk
        high_risk_flags = []
        if yes_pct < 50:
            high_risk_flags.append("Low_ES_Contribution")
        if afp_faster_pct > 50:
            high_risk_flags.append("AFP_Faster_than_ES")
        if avg_challenges > overall_chal_mean:
            high_risk_flags.append("High_Challenges")

        high_risk_summary.append({
            "State": state,
            "ES_Contribution_%": round(yes_pct,2),
            "AFP_Faster_%": round(afp_faster_pct,2),
            "Avg_Num_Challenges": round(avg_challenges,2),
            "High_Risk_Flags": ", ".join(high_risk_flags) if high_risk_flags else "None"
        })

high_risk_df = pd.DataFrame(high_risk_summary).sort_values(by="ES_Contribution_%")
print("High-risk state summary:")
print(high_risk_df)

# Save to Excel
high_risk_df.to_excel(risk_writer, sheet_name="High_Risk_States", index=False)
risk_writer.close()
print("High-risk state summary saved to:", EXTRA_XLSX)


# HIGH-RISK STATE HEATMAP

print("\n==== HIGH-RISK STATE HEATMAP ====\n")

if not high_risk_df.empty:
    # Prepare binary flags for heatmap
    flags = ["Low_ES_Contribution", "AFP_Faster_than_ES", "High_Challenges"]
    heatmap_df = pd.DataFrame(0, index=high_risk_df["State"], columns=flags)
    
    for idx, row in high_risk_df.iterrows():
        for f in flags:
            if f in row["High_Risk_Flags"]:
                heatmap_df.at[row["State"], f] = 1

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(8, max(4, len(heatmap_df)//2)))
    sns.heatmap(heatmap_df, annot=True, cmap="Reds", cbar=False, linewidths=0.5, linecolor="gray")
    ax.set_title("High-Risk Flags by State")
    ax.set_ylabel("State")
    ax.set_xlabel("High-Risk Flag")
    plt.tight_layout()
    save_png(fig, "high_risk_state_heatmap")
    print("High-risk heatmap saved to plots folder.")
else:
    print("No high-risk state data available for heatmap.")



# EXTRA ANALYSES

print("\n==== EXTRA ANALYSES ====\n")
extra_tables = {}
# num challenges distribution
if challenge_bin_cols:
    df["num_challenges"] = df[challenge_bin_cols].sum(axis=1)
    ntab = freq_table(df["num_challenges"].fillna(0))
    print_table(ntab, "Number of challenges selected")
    extra_tables["num_challenges"] = ntab
    fig, ax = plt.subplots(figsize=(7,4))
    sns.histplot(df["num_challenges"].dropna(), bins=range(0, int(df["num_challenges"].max())+2), ax=ax)
    ax.set_xlabel("Number of challenges selected")
    save_png(fig, "num_challenges_hist")
# correct definition percent (option 2)
def_map = mapping_info.get("What is environmental surveillance about?")
if def_map:
    def_lab = def_map["label_col"]
    correct = CODEBOOK["What is environmental surveillance about?"][2]
    pct = (df[def_lab] == correct).mean() * 100
    print(f"Proportion selecting the 'monitoring hazards' definition: {pct:.2f}%")
    extra_tables["correct_definition_pct"] = pd.DataFrame({"Correct_definition_percent":[round(pct,2)], "n":[len(df)]})
    extra_tables["es_def_distribution"] = freq_table(df[def_lab].fillna("Missing"))
with pd.ExcelWriter(EXTRA_XLSX, engine="openpyxl", mode="w") as w:
    write_excel_sheets(w, extra_tables)
print("Extra analyses saved to:", EXTRA_XLSX)


# INTERPRETATION & RECOMMENDATIONS

recfile = os.path.join(TEXT_DIR, "interpretation_recommendations.txt")
with open(recfile, "w", encoding="utf-8") as f:
    f.write("Interpretation & Recommendations\n")
    f.write("="*60 + "\n\n")
    f.write("See Excel tables in outputs/tables and figures in outputs/plots.\n")
    f.write("- Objective 1: check 'ES_contributed' descriptive table and inferential sheets '..._vs_EScontrib'.\n")
    f.write("- Objective 2: check 'Effectiveness_by_state' and 'Effectiveness_vs_WhichFaster' sheets.\n")
    f.write("- Objective 3: challenges summary and 'Challenges_vs_EScontrib' sheet for associations.\n\n")
print("Interpretation notes saved to:", recfile)

print("\nALL DONE. Key outputs:")
print(" - Descriptive Excel:", DESCRIPTIVE_XLSX)
print(" - Inferential Excel:", INFERENTIAL_XLSX)
print(" - Model Excel:", MODELS_XLSX)
print(" - Extra Excel:", EXTRA_XLSX)
print(" - Plots folder:", PLOTS_DIR)
print(" - Cleaned CSV:", cleaned_csv)
