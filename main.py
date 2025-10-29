import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ---------------------------
# Step 0: Setup NLTK
# ---------------------------
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# ---------------------------
# Step 1: Load datasets safely
# ---------------------------
def safe_load(file):
    try:
        df = pd.read_csv(file)
        print(f"✅ Loaded {file} | Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"⚠️ Could not load {file}: {e}")
        return pd.DataFrame()

data1 = safe_load("train.csv")
data2 = safe_load("Mental_Health_FAQ.csv")
data3 = safe_load("Combined Data.csv")

# ---------------------------
# Step 2: Normalize and standardize columns
# ---------------------------
datasets = [data1, data2, data3]
for df in datasets:
    df.columns = [c.lower().strip() for c in df.columns]

def standardize(df):
    if len(df) == 0:
        return pd.DataFrame(columns=["text", "label"])

    # detect likely text and label columns
    text_col = next((c for c in df.columns if 'text' in c or 'content' in c or 'question' in c), None)
    label_col = next((c for c in df.columns if 'label' in c or 'target' in c or 'response' in c or 'answer' in c or 'emotion' in c), None)

    if not text_col:
        print("⚠️ No text column found.")
        return pd.DataFrame(columns=["text", "label"])

    if not label_col:
        print("⚠️ No label column found. Assigning dummy Neutral labels.")
        df['label'] = 'Neutral'
    else:
        df = df.rename(columns={text_col: 'text', label_col: 'label'})

    df = df[['text', 'label']].dropna()
    return df

data1 = standardize(data1)
data2 = standardize(data2)
data3 = standardize(data3)

combined = pd.concat([data1, data2, data3], ignore_index=True)
print(f"\n📊 Combined dataset shape: {combined.shape}")

# ---------------------------
# Step 3: Normalize labels
# ---------------------------
label_map = {
    "suicide": "Negative",
    "non-suicide": "Positive",
    "negative": "Negative",
    "positive": "Positive",
    "neutral": "Neutral",
}
combined['label'] = combined['label'].astype(str).str.lower().map(label_map).fillna(combined['label'])

# Ensure at least 2 classes
if combined['label'].nunique() < 2:
    print("⚠️ Only one class found, creating synthetic labels.")
    half = len(combined) // 2
    combined.loc[:half, 'label'] = "Positive"
    combined.loc[half:, 'label'] = "Negative"

print("\n📊 Label distribution:")
print(combined['label'].value_counts())


# Normalize labels
def normalize_label(label):
    label = str(label).lower().strip()
    if any(w in label for w in ["suicid", "depress", "sad", "angry", "fear", "hate", "anxious", "worry", "stress"]):
        return "Negative"
    elif any(w in label for w in ["happy", "joy", "grateful", "love", "calm", "great", "good"]):
        return "Positive"
    elif "neutral" in label or label == "":
        return "Neutral"
    else:
        return "Neutral"

combined["label"] = combined["label"].apply(normalize_label)
print("✅ Label distribution after normalization:")
print(combined["label"].value_counts())

# ---------------------------
# Step 4: Clean text
# ---------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = [stemmer.stem(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

combined["clean_text"] = combined["text"].apply(clean_text)

# ---------------------------
# Step 5: Remove rare labels
# ---------------------------
# Remove rare labels
label_counts = combined["label"].value_counts()
valid_labels = label_counts[label_counts > 5].index
filtered_data = combined[combined["label"].isin(valid_labels)]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    filtered_data["clean_text"], filtered_data["label"],
    test_size=0.2, random_state=42, stratify=filtered_data["label"]
)

# TF-IDF + Model
vectorizer = TfidfVectorizer(max_features=7000, ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluation
y_pred = model.predict(X_test_vec)
print("\n✅ Improved Model Report:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
print("\n🎉 Improved model saved successfully!")

