import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import numpy as np
from scipy.stats import zscore  # Outlier detection

csv_path = "/Users/ismailperacha/Desktop/Amazon_Initial.csv"
df = pd.read_csv(csv_path, skiprows=1)

print("Data loaded successfully!")
print("Initial Shape:", df.shape)
print(df.head())

# Dropping completely empty columns and duplicate rows
df.dropna(axis=1, how='all', inplace=True)
df.drop_duplicates(inplace=True)

print("Cleaned Shape:", df.shape)

# Visualizing label distribution (if label exists)
if 'label' in df.columns:
    df['label'].value_counts().plot(kind='bar', title='Class Distribution (0 = Real, 1 = Fake)')
    plt.xlabel("Review Type")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

# Identifying non-feature columns here
non_feature_cols = ['id', 'label'] if 'label' in df.columns else ['id']
feature_cols = df.columns.difference(non_feature_cols)
df.fillna(df.median(numeric_only=True), inplace=True)

# Removing any rows where all Bag of Words values are zero
df = df[df[feature_cols].sum(axis=1) > 0]
print("Shape after removing empty rows:", df.shape)
# Remove outliers using Z-score
from scipy.stats import zscore
z_scores = np.abs(zscore(df[feature_cols]))
filtered_df = df[(z_scores < 3).all(axis=1)]
if not filtered_df.empty:
    df = filtered_df
else:
    print("Warning: Z-score filtering removed all rows. Skipping outlier removal.")
# now we can look for correlations
print(df.describe())
print(df.info())
# Example feature distribution
df[feature_cols].sample(n=5, axis=1).hist(bins=50, figsize=(15, 10))
plt.suptitle("Feature Distributions")
plt.tight_layout()
plt.show()

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[feature_cols])

# Dimensionality Reduction using TruncatedSVD (and PCA for sparse data)
svd = TruncatedSVD(n_components=100, random_state=42)
reduced_features = svd.fit_transform(scaled_features)

# Creating new DataFrame from the SVD features
df_processed = pd.DataFrame(reduced_features, columns=[f"component_{i}" for i in range(1, 101)])

# Reattaching label if it was in the original dataset
if 'label' in df.columns:
    df_processed['label'] = df['label'].values

# Saving processed data
output_path = "/Users/ismailperacha/Desktop/Amazon_Processed2.csv"
df_processed.to_csv(output_path, index=False)

print("Preprocessing complete. File saved as 'Amazon_Processed2.csv'")
print(df_processed.shape)
print(df_processed.columns.tolist())
print(df_processed.head())

print(df_processed.columns)
