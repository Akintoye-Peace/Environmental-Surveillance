import pandas as pd
import matplotlib.pyplot as plt

# Dataset
data = {
    "State": ["Abia","Anambra","Bauchi","Borno","Delta","Gombe","Jigawa","Kaduna",
              "Kano","Katsina","Kebbi","Kogi","Kwara","Niger","Ogun","Osun",
              "Oyo","Rivers","Sokoto","Yobe","Zamfara"],
    "No": [21.43, 80.00, 0.00, 0.00, 54.84, 0.00, 100.00, 0.00, 0.00, 33.33,
           10.00, 28.57, 0.00, 100.00, 46.67, 47.06, 70.27, 0.00, 22.22, 5.26, 3.70],
    "Not_Sure": [57.14, 10.00, 100.00, 0.00, 19.35, 0.00, 0.00, 0.00, 0.00, 16.67,
                 5.00, 71.43, 50.00, 0.00, 33.33, 23.53, 18.92, 100.00, 22.22, 15.79, 0.00],
    "Yes": [21.43, 10.00, 0.00, 100.00, 25.81, 100.00, 0.00, 100.00, 100.00, 50.00,
            85.00, 0.00, 50.00, 0.00, 20.00, 29.41, 10.81, 0.00, 55.56, 78.95, 96.30]
}

df = pd.DataFrame(data)
df.set_index("State", inplace=True)

# Plot
fig, ax = plt.subplots(figsize=(16, 9))
colors = ["#FF6F61", "#FFD700", "#4CAF50"]  # No = red, Not Sure = yellow, Yes = green
df.plot(kind="bar", stacked=True, color=colors, ax=ax)

# Labels, title, legend
ax.set_ylabel("Percentage (%)", fontsize=14)
ax.set_xlabel("State", fontsize=14)
ax.set_title("State-wise Environmental Surveillance (ES) Contribution to Early Detection of Poliovirus", fontsize=16)
ax.tick_params(axis='x', rotation=45, labelsize=12)
ax.tick_params(axis='y', labelsize=12)
ax.legend(title="Response", fontsize=12, title_fontsize=12, bbox_to_anchor=(1.05, 1), loc='upper left')

# Add value labels, handle small percentages
for i, row in enumerate(df.itertuples()):
    # "No" bar
    y_no = row.No / 2 if row.No >= 2 else row.No + 1
    ax.text(i, y_no, f"{row.No:.1f}%", ha='center', va='center', color='white' if row.No >= 2 else 'black', fontsize=11)
    
    # "Not_Sure" bar
    y_ns = row.No + row.Not_Sure / 2 if row.Not_Sure >= 2 else row.No + row.Not_Sure + 1
    ax.text(i, y_ns, f"{row.Not_Sure:.1f}%", ha='center', va='center', color='black', fontsize=11)
    
    # "Yes" bar
    y_yes = row.No + row.Not_Sure + row.Yes / 2 if row.Yes >= 2 else row.No + row.Not_Sure + row.Yes + 1
    ax.text(i, y_yes, f"{row.Yes:.1f}%", ha='center', va='center', color='white' if row.Yes >= 2 else 'black', fontsize=11)

plt.tight_layout()
plt.savefig("Statewise_ES_Contribution.png", dpi=300)
plt.show()
