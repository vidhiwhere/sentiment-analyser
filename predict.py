import joblib
import re


model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)            
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()         
    return text


def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]

   
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        confidence = round(max(probs) * 100, 2)
        return f"{pred} ({confidence}% confidence)"
    else:
        return pred

if __name__ == "__main__":
    print("✨ Sentiment Predictor ✨")
    print("Type your text below (or 'q' to quit):\n")

    while True:
        user_input = input("Enter text: ")
        if user_input.lower() == "q":
            print("\n👋 Exiting Sentiment Predictor. Goodbye!")
            break

        result = predict_sentiment(user_input)
        print("Predicted Sentiment:", result)
        print("-" * 50)
