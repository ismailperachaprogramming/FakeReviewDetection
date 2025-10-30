import pandas as pd
from sklearn.utils import shuffle

# Load both processed datasets
df_fake = pd.read_csv("/Users/ismailperacha/Downloads/fake_reviews_processed1.csv")
df_amazon = pd.read_csv("/Users/ismailperacha/Desktop/Amazon_Processed2.csv")

#check shape & colums
print(df_fake.shape, df_amazon.shape)
print(df_fake.columns)
print(df_amazon.columns)

#concatenate data
df_combined = pd.concat([df_fake, df_amazon], ignore_index=True)

#shuffle data 
df_combined = shuffle(df_combined, random_state=42)

#save data 
df_combined.to_csv("/Users/ismailperacha/Desktop/combined_dataset.csv", index=False)
print("Combined dataset saved! ✅")

import matplotlib.pyplot as plt
import seaborn as sns

# # Compute correlation matrix (excluding label)
# correlation_matrix = df_combined.drop("label", axis=1).corr()

# plt.figure(figsize=(12, 10))
# sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True,
#             cbar_kws={"shrink": .5}, xticklabels=False, yticklabels=False)

# plt.title("Combined datasets: Correlation Heatmap of Features")
# plt.tight_layout()
# plt.show()

# Correlation with label only
# label_corr = df_combined.corr()["label"].drop("label")

plt.figure(figsize=(10, 10))  # Taller plot for better spacing
label_corr.sort_values().plot(kind="barh")
plt.title("Feature Correlation with Label", fontsize=12)
plt.xlabel("Correlation",fontsize=10)
plt.yticks(fontsize=5)  # 👈adjust smaller font for y-axis labels
plt.tight_layout()
plt.show()
