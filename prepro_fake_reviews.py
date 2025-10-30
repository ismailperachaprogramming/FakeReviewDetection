
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

# Load data
df = pd.read_csv(r"C:\Users\17147\OneDrive\Desktop\ML Project Data\fake reviews dataset.csv")
print("Initial shape:", df.shape)

# Drop duplicates and missing
df.drop_duplicates(inplace=True)
df.dropna(inplace=True) 
print("Cleaned shape:", df.shape)

# Optional: Visualize label distribution
df['label'].value_counts().plot(kind='bar', title='Label Distribution')
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Encode label column (e.g., CG = 1, OR = 0)
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

# Optional engineered features (uncomment to include)
# df['review_length'] = df['text_'].apply(len)
# df['num_exclamations'] = df['text_'].str.count('!')

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_tfidf = vectorizer.fit_transform(df['text_'])

# Dimensionality Reduction using TruncatedSVD
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)

# Combine reduced components into DataFrame
df_final = pd.DataFrame(X_reduced, columns=[f"component_{i}" for i in range(1, 101)])
df_final['label'] = df['label_encoded'].values

# Optional: Add engineered features to final DataFrame
# df_final['review_length'] = df['review_length'].values
# df_final['num_exclamations'] = df['num_exclamations'].values

# Correlation Heatmap (based on reduced components)
plt.figure(figsize=(12, 10))
sns.heatmap(df_final.corr(), cmap='coolwarm')
plt.title("Correlation Heatmap of Components")
plt.tight_layout()
plt.show()

# Save processed data
df_final.to_csv(r"C:\Users\17147\OneDrive\Desktop\ML Project Data\fake_reviews_processed1.csv", index=False)
print("Preprocessing complete. File saved as 'fake_reviews_processed1.csv'")
print(df_final.shape)
print(df_final.columns.tolist())
print(df_final.head())
