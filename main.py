import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from joblib import Parallel, delayed


nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


try:
 
    data2 = pd.read_csv("training.1600000.processed.noemoticon.csv", encoding='latin-1', header=None, nrows=100000)
    data2.columns = ['target', 'ids', 'date', 'flag', 'user', 'text']
    data2 = data2[['text', 'target']]
    data2['target'] = data2['target'].replace({0: 'Negative', 4: 'Positive'})
    print("data2 target values:", data2['target'].value_counts())

   
    data1 = pd.read_csv("train.csv")
    print("data1 columns:", data1.columns)
    data1.columns = data1.columns.str.strip().str.lower()  # Normalize column names
    data1 = data1.rename(columns={'context': 'text', 'response': 'target'})  # Use lowercase
    print("data1 columns after rename:", data1.columns)
    if 'target' not in data1.columns:
        print("Error: 'target' column not found in data1. Available columns:", data1.columns)
        exit()
    print("data1 target values:", data1['target'].value_counts())

    
    data3 = pd.read_csv("Suicide_Detection.csv")
    print("data3 columns:", data3.columns)
    data3 = data3.rename(columns={'content': 'text', 'emotion': 'target'})
    data3['target'] = data3['target'].map({'suicide': 'Negative', 'non-suicide': 'Positive'})
    print("data3 target values:", data3['target'].value_counts())

    data4 = pd.read_csv("Mental_Health_FAQ.csv")
    print("data4 columns:", data4.columns)
    data4.rename(columns={'question': 'text', 'answer': 'target'}, inplace=True)

   
    data5 = pd.read_csv("Combined Data.csv")
    print("data5 columns:", data5.columns)
    data5 = data5.rename(columns={'text_data': 'text', 'sentiment_label': 'target'})
    print("data5 target values:", data5['target'].value_counts())

except (FileNotFoundError, KeyError) as e:
    print(f"Error: {e}. Check file paths or column names.")
    exit()

print(data1.columns)
print(data2.columns)
print(data3.columns)
print(data4.columns)
print(data5.columns)
print(data2.head())


combined = pd.concat([data1, data2, data3,data4, data5], axis=0, ignore_index=True)
combined = combined.dropna(subset=['text', 'target']).reset_index(drop=True)
print("Combined dataset shape:", combined.shape)
print(combined.head())
print("Unique target values:", combined['target'].value_counts())


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
    return text


combined = combined.sample(frac=0.1, random_state=42)
combined['clean_text'] = Parallel(n_jobs=-1)(delayed(clean_text)(text) for text in combined['text'])


X = combined['clean_text']
y = combined['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer(max_features=2000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression(max_iter=500)
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

sample = ["I feel really anxious today", "I am happy and relaxed"]
sample_vec = vectorizer.transform(sample)
print("Sample predictions:", model.predict(sample_vec))