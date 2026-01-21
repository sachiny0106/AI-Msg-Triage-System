import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

def train():
    # 1. Load dataset
    print("Loading dataset...")
    try:
        df = pd.read_csv("data/messages.csv")
    except FileNotFoundError:
        print("Error: data/messages.csv not found. Run generate_data.py first.")
        return

    print(f"Loaded {len(df)} examples.")

    # 2. Split into train/validation (80/20, stratified)
    X = df["text"]
    y = df["label"]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3. Pipeline: TF-IDF + LogisticRegression
    # Using min_df=2 as requested. Unigrams and bigrams.
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000))
    ])

    # 4. Train the model
    print("Training model...")
    pipeline.fit(X_train, y_train)

    # 5. Evaluate
    print("Evaluating...")
    y_pred = pipeline.predict(X_val)
    
    # Per-class metrics
    report = classification_report(y_val, y_pred)
    print(report)

    # Macro F1
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    print(f"Macro F1: {macro_f1:.2f}")

    # 6. Save artifacts
    print("Saving artifacts...")
    joblib.dump(pipeline, "models/model.joblib")
    
    # Save vectorizer separately for convenience
    vectorizer = pipeline.named_steps["tfidf"]
    joblib.dump(vectorizer, "models/vectorizer.joblib")

    print("Done.")

if __name__ == "__main__":
    train()
