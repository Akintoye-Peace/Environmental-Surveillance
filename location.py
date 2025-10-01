import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Data ---
data = {
    "Location": [
        "Zamfara", "Sokoto", "Kebbi", "Yobe", "Oyo", "Osun", "Delta", "Ogun",
        "Abia", "Anambra", "Kano", "Kaduna", "Kogi", "Katsina", "Kwara",
        "Gombe", "Bauchi", "Niger", "Jigawa", "Rivers", "Borno"
    ],
    "Count": [
        54, 54, 40, 38, 37, 34, 31, 30, 28, 20, 14, 14, 14,
        12, 12, 6, 4, 2, 2, 2, 2
    ],
    "Percent": [
        12, 12, 8.89, 8.44, 8.22, 7.56, 6.89, 6.67, 6.22, 4.44,
        3.11, 3.11, 3.11, 2.67, 2.67, 1.33, 0.89, 0.44, 0.44, 0.44, 0.44
    ]
}

df = pd.DataFrame(data)

# --- Sort by count (descending) for better readability ---
df = df.sort_values(by="Count", ascending=True)

# --- Visualization ---
plt.figure(figsize=(8, 10))
bars = plt.barh(df["Location"], df["Count"], color="steelblue")

# Annotate counts on bars
for bar, count, pct in zip(bars, df["Count"], df["Percent"]):
    plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{count} ({pct:.2f}%)", va="center", fontsize=9)

plt.xlabel("Number of Respondents")
plt.title("Location/State of Primary Assignment (N = 450)", fontsize=12, weight="bold")
plt.tight_layout()

# --- Save output ---
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)
plt.savefig(os.path.join(output_folder, "location_distribution.png"), dpi=300)
plt.close()

print("Chart saved to:", os.path.join(output_folder, "location_distribution.png"))
