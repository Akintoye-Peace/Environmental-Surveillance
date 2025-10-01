import pandas as pd
import os
from collections import defaultdict

# ----------------------------
# File paths
# ----------------------------
INPUT_PATH = r"C:\Users\everp\Documents\Surveillance\surveillance.csv"
OUTPUT_RECODED = r"C:\Users\everp\Documents\Surveillance\surveillance_recoded.csv"
OUTPUT_CODEBOOK = r"C:\Users\everp\Documents\Surveillance\codebook.csv"

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(INPUT_PATH)

# ----------------------------
# List of Nigerian States (36 + FCT)
# ----------------------------
nigerian_states = [
    "Abia","Adamawa","Akwa Ibom","Anambra","Bauchi","Bayelsa","Benue","Borno",
    "Cross River","Delta","Ebonyi","Edo","Ekiti","Enugu","FCT","Gombe","Imo",
    "Jigawa","Kaduna","Kano","Katsina","Kebbi","Kogi","Kwara","Lagos","Nasarawa",
    "Niger","Ogun","Ondo","Osun","Oyo","Plateau","Rivers","Sokoto","Taraba",
    "Yobe","Zamfara"
]

# Create mapping dictionary
state_map = {state: i+1 for i, state in enumerate(nigerian_states)}

# ----------------------------
# Define recoding dictionaries for other categorical variables
# ----------------------------
recode_maps = {
    "Gender": {"Male": 1, "Female": 2},
    "Age Group": {"18-'30": 1, "31-'40": 2, "41-'50": 3, "51+": 4},
    "Current Position": {
        "Surveillance Officer": 1,
        "Laboratory Scientist": 2,
        "Environmental Health Officer": 3,
        "Health Facility Worker": 4,
        "Other…": 5
    },
    "Primary Role in Environmental Surveillance for Polio Eradication": {
        "Sample collection": 1,
        "Laboratory analysis": 2,
        "Data management and reporting": 3,
        "Supervision/Coordination": 4,
        "Routine Inspection of Premises": 5,
        "Other…": 6
    },
    "Years of Experience ": {
        "Less than 5 years": 1,
        "5-'10 years": 2,
        "11-'15 years": 3,
        "Over 15 years": 4
    },
    "Are you aware of environmental surveillance (ES) activities in your area? ": {
        "Yes": 1,
        "No": 0
    },
    "How familiar are you with the processes of ES (sewage sampling, lab testing)?": {
        "Very familiar": 3,
        "Somewhat familiar": 2,
        "Not familiar": 1
    },
    "What is environmental surveillance about?": {
        "Tracking wildlife populations to maintain ecological balance and biodiversity": 1,
        "Monitoring and analyzing environmental hazards (e.g., pollution, toxins) to prevent health risks and guide public health actions": 2,
        "Enforcing workplace safety regulations to prevent occupational injuries": 3,
        "Studying genetic diseases to develop personalized medical treatments": 4
    },
    "Have you received formal training on ES? ": {"Yes": 1, "No": 0},
    "How effective is ES in detecting poliovirus compared to  Acute Flaccid Paralysis (AFP) ?": {
        "More effective": 3, "Equally effective": 2, "Less effective": 1, "Don't know": 0
    },
    "Has ES in your area contributed to early detection of poliovirus cases? ": {
        "Yes": 1, "No": 0, "Not sure": 2
    },
    "ES has played a role in polio eradication in Nigeria": {
        "Strongly agree": 5, "Agree": 4, "Neutral": 3, "Disagree": 2, "Strongly disagree": 1
    },
    "Frequency of environmental sample collection": {
        "Weekly": 4, "Bi-weekly": 3, "Monthly": 2, "Rarely": 1, "Not applicable": 0
    },
    "Do you work with both AFP and ES systems? ": {"Yes": 1, "No": 0},
    "Which system has faster virus detection?": {"ES": 1, "AFP": 2, "Both": 3, "Not sure": 0}
}

# ----------------------------
# Recode categorical variables
# ----------------------------
codebook_records = []
df_recoded = df.copy()

for col, mapping in recode_maps.items():
    if col in df.columns:
        df_recoded[col] = df[col].map(mapping)
        for k, v in mapping.items():
            codebook_records.append([col, k, v])

# ----------------------------
# Recode State Variable
# ----------------------------
if "Location/State of Primary Assignment" in df.columns:
    df_recoded["Location/State of Primary Assignment"] = df["Location/State of Primary Assignment"].map(state_map)
    for state, code in state_map.items():
        codebook_records.append(["Location/State of Primary Assignment", state, code])

# ----------------------------
# Handle Multiple Response Question (Challenges)
# ----------------------------
challenge_options = [
    "Poor funding","Inadequate training","Sample collection difficulties",
    "Delay in lab analysis","Insecurity in sampling areas",
    "Logistical/transport issues","Lack of community awareness","Other…"
]

if "Challenges encountered in ES (check all that apply)" in df.columns:
    for option in challenge_options:
        new_col = f"Challenge_{option.replace(' ', '_')}"
        df_recoded[new_col] = df["Challenges encountered in ES (check all that apply)"].apply(
            lambda x: 1 if pd.notna(x) and option in str(x) else 0
        )
        codebook_records.append(["Challenges encountered in ES", option, f"Binary {new_col} (1=Selected, 0=Not selected)"])
    df_recoded.drop(columns=["Challenges encountered in ES (check all that apply)"], inplace=True)

# ----------------------------
# Save recoded dataset
# ----------------------------
df_recoded.to_csv(OUTPUT_RECODED, index=False)

# ----------------------------
# Save codebook
# ----------------------------
codebook_df = pd.DataFrame(codebook_records, columns=["Variable", "Original Value", "Numeric Code"])
codebook_df.to_csv(OUTPUT_CODEBOOK, index=False)

print("✅ Recoding complete. Files saved:")
print(f"- Recoded dataset: {OUTPUT_RECODED}")
print(f"- Codebook: {OUTPUT_CODEBOOK}")
