import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

print("[INFO] Generating EDA plots...")
df = pd.read_csv("data/eda_ready_data.csv")
os.makedirs("plots", exist_ok=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.histplot(df["Price_in_Lakhs"], bins=30, ax=axes[0, 0], color="teal").set_title("Q1: Price Distribution")
sns.histplot(df["Size_in_SqFt"], bins=30, ax=axes[0, 1], color="coral").set_title("Q2: Size Distribution")
sns.boxplot(data=df, x="Property_Type", y="Price_per_SqFt", ax=axes[1, 0]).set_title("Q3 & Q5: Price/SqFt by Type")
sns.scatterplot(data=df.sample(min(1500, len(df)), random_state=42), x="Size_in_SqFt", y="Price_in_Lakhs", ax=axes[1, 1]).set_title("Q4: Size vs Price")
plt.tight_layout()
plt.savefig("plots/1_price_and_size_analysis.png")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
s_pps = df.groupby("State")["Price_per_SqFt"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=s_pps.values, y=s_pps.index, ax=axes[0, 0]).set_title("Q6: Top 10 States by Price/SqFt")
c_p = df.groupby("City")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(10)
sns.barplot(x=c_p.values, y=c_p.index, ax=axes[0, 1]).set_title("Q7: Top 10 Cities by Price")
top_loc = df.groupby("Locality")["Price_in_Lakhs"].mean().sort_values(ascending=False).head(5).index
sns.barplot(data=df[df["Locality"].isin(top_loc)], x="Locality", y="Price_in_Lakhs", ax=axes[1, 0]).set_title("Q8 & Q10: Top 5 Localities")
top_c = df["City"].value_counts().head(5).index
sns.countplot(data=df[df["City"].isin(top_c)], x="City", hue="BHK", ax=axes[1, 1]).set_title("Q9: BHK by City")
plt.tight_layout()
plt.savefig("plots/2_location_based_analysis.png")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
num_df = df.select_dtypes(include=[np.number])
sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0, 0]).set_title("Q11: Correlation Heatmap")
sns.barplot(data=df, x="Nearby_Schools", y="Price_per_SqFt", ax=axes[0, 1]).set_title("Q12 & Q13: Schools vs Price/SqFt")
sns.boxplot(data=df, x="Furnished_Status", y="Price_in_Lakhs", ax=axes[1, 0]).set_title("Q14: Price by Furnished Status")
sns.barplot(data=df, x="Facing", y="Price_per_SqFt", ax=axes[1, 1]).set_title("Q15: Price/SqFt by Facing")
plt.tight_layout()
plt.savefig("plots/3_feature_correlations.png")
plt.close()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
df["Owner_Type"].value_counts().plot.pie(autopct="%1.1f%%", ax=axes[0, 0]).set_title("Q16: Owner Type")
sns.countplot(data=df, x="Availability_Status", hue="Good_Investment", ax=axes[0, 1]).set_title("Q17: Availability Status")
sns.barplot(data=df, x="Parking_Space", y="Price_in_Lakhs", ax=axes[1, 0]).set_title("Q18: Parking vs Price")
sns.countplot(data=df, x="Public_Transport_Accessibility", hue="Good_Investment", ax=axes[1, 1]).set_title("Q19 & Q20: Transport vs Investment")
plt.tight_layout()
plt.savefig("plots/4_investment_and_amenities.png")
plt.close()
print("[SUCCESS] All EDA plots saved!")
