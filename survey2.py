# survey1.py
# Comprehensive Analysis of Environmental Surveillance Survey in Nigeria
# Author: Everpeace Research Institute
# --------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from scipy.stats import chi2_contingency
import statsmodels.formula.api as smf

# --------------------------------------------------------------
# SETUP
# --------------------------------------------------------------
os.makedirs("plots", exist_ok=True)
os.makedirs("excel_outputs", exist_ok=True)

# --------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------
df_raw = pd.read_csv("surveillance_recoded.csv")
print(f"Loaded data with shape: {df_raw.shape}")

df_raw.columns = df_raw.columns.str.strip()

# --------------------------------------------------------------
# CODEBOOK (mappings)
# --------------------------------------------------------------
codebook = {
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
    "Are you aware of environmental surveillance (ES) activities in your area?": {
        1: "Yes", 0: "No"
    },
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
    "Has ES in your area contributed to early detection of poliovirus cases?": {
        1: "Yes", 0: "No", 2: "Not sure"
    },
    "ES has played a role in polio eradication in Nigeria": {
        5: "Strongly agree", 4: "Agree", 3: "Neutral", 2: "Disagree", 1: "Strongly disagree"
    },
    "Frequency of environmental sample collection": {
        4: "Weekly", 3: "Bi-weekly", 2: "Monthly", 1: "Rarely", 0: "Not applicable"
    },
    "Do you work with both AFP and ES systems?": {1: "Yes", 0: "No"},
    "Which system has faster virus detection?": {
        1: "AFP", 2: "ES", 3: "Both", 4: "Not sure"
    }
}

# Apply mappings
for col, mapping in codebook.items():
    if col in df_raw.columns:
        df_raw[col] = df_raw[col].map(mapping).fillna(df_raw[col])

# --------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------
def safe_filename(name):
    return re.sub(r"[^\w]", "_", str(name))[:50]

def save_table(df, name):
    fname = safe_filename(name)
    df.to_excel(f"excel_outputs/{fname}.xlsx")

def save_barplot(series, title, fname, horizontal=False):
    counts = series.value_counts(dropna=False)
    fname = safe_filename(fname)
    plt.figure(figsize=(8, 5))
    if horizontal:
        sns.barplot(y=counts.index, x=counts.values, palette="viridis")
        plt.xlabel("Count")
        plt.ylabel("")
    else:
        sns.barplot(x=counts.index, y=counts.values, palette="viridis")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"plots/{fname}.png", dpi=300)
    plt.close()

def chi2_test(x, y, name):
    ctab = pd.crosstab(x, y)
    if ctab.shape[0] > 1 and ctab.shape[1] > 1:
        chi2, p, dof, exp = chi2_contingency(ctab)
        result = pd.DataFrame({"Chi2": [chi2], "p-value": [p], "df": [dof]})
        save_table(ctab, f"CrossTab_{name}")
        save_table(result, f"Chi2_{name}")
        return result
    return None

def logistic_model(df, outcome, predictors, name):
    formula = f"{outcome} ~ " + " + ".join(predictors)
    try:
        model = smf.logit(formula=formula, data=df).fit(disp=False)
        summary = model.summary2().tables[1]
        save_table(summary, f"Logit_{name}")
    except Exception as e:
        print(f"Logistic regression failed for {name}: {e}")

# --------------------------------------------------------------
# ANALYSIS
# --------------------------------------------------------------

### Objective 1: Early detection
print("\nObjective 1: Contribution of ES to early detection")
col = "Has ES in your area contributed to early detection of poliovirus cases?"
if col in df_raw:
    save_barplot(df_raw[col], "ES contribution to early detection", col)
    chi2_test(df_raw[col], df_raw.get("Have you received formal training on ES?", pd.Series()), "EarlyDetection_vs_Training")

### Objective 2: ES vs AFP
print("\nObjective 2: ES vs AFP Comparison")
col = "How effective is ES in detecting poliovirus compared to  Acute Flaccid Paralysis (AFP) ?"
if col in df_raw:
    save_barplot(df_raw[col], "Effectiveness of ES vs AFP", col)
    chi2_test(df_raw[col], df_raw.get("Do you work with both AFP and ES systems?", pd.Series()), "ESvsAFP_vs_WorkWithBoth")

### Objective 3: Challenges
print("\nObjective 3: Operational Challenges")
challenges = [c for c in df_raw.columns if "Challenge_" in c]
if challenges:
    challenge_counts = df_raw[challenges].sum().sort_values(ascending=False)
    challenge_df = pd.DataFrame({"Challenge": challenge_counts.index, "Count": challenge_counts.values})
    save_table(challenge_df, "Challenges_Summary")
    save_barplot(challenge_counts, "Challenges in ES", "Challenges", horizontal=True)

### Objective 4: Strategies
print("\nObjective 4: Strategies to Improve")
chi2_test(df_raw.get("Are you aware of environmental surveillance (ES) activities in your area?", pd.Series()),
          df_raw.get("Have you received formal training on ES?", pd.Series()),
          "Awareness_vs_Training")

chi2_test(df_raw.get("What is environmental surveillance about?", pd.Series()),
          df_raw.get("Have you received formal training on ES?", pd.Series()),
          "Knowledge_vs_Training")

# Logistic regression: Awareness ~ Training + Experience + Position
if "Are you aware of environmental surveillance (ES) activities in your area?" in df_raw:
    df_model = df_raw.dropna(subset=["Are you aware of environmental surveillance (ES) activities in your area?"]).copy()
    df_model["Awareness_bin"] = (df_model["Are you aware of environmental surveillance (ES) activities in your area?"] == "Yes").astype(int)
    df_model["Training_bin"] = (df_model.get("Have you received formal training on ES?", pd.Series()) == "Yes").astype(int)
    logistic_model(df_model,
                   "Awareness_bin",
                   ["Training_bin", "C(`Years of Experience`)", "C(`Current Position`)"],
                   "Awareness_Model")

### State-level analysis
print("\nState-level analysis")
if "Location/State of Primary Assignment" in df_raw:
    state_awareness = pd.crosstab(df_raw["Location/State of Primary Assignment"],
                                  df_raw["Are you aware of environmental surveillance (ES) activities in your area?"],
                                  normalize="index") * 100
    save_table(state_awareness, "State_Awareness")
    state_awareness.plot(kind="bar", stacked=True, figsize=(12, 7), colormap="viridis")
    plt.title("Awareness of ES by State (%)")
    plt.ylabel("Percentage")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("plots/state_awareness.png", dpi=300)
    plt.close()

print("\n✅ Analysis complete. Results saved in /plots and /excel_outputs")
