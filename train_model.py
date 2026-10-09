from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
FAKE_PATH = BASE_DIR / "Fake.csv"
REAL_PATH = BASE_DIR / "True.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "fake_news_model.pkl"


def find_column(df, expected_name):
    for column in df.columns:
        if column.strip().lower() == expected_name.lower():
            return column
    raise ValueError(
        f"Required column '{expected_name}' was not found. "
        f"Available columns: {list(df.columns)}"
    )


def read_news(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path.name}. "
            "Download the dataset described in README.md and place "
            "Fake.csv and True.csv beside train_model.py."
        )

    df = pd.read_csv(path)
    title_col = find_column(df, "title")
    text_col = find_column(df, "text")

    title = df[title_col].fillna("").astype(str)
    article = df[text_col].fillna("").astype(str)
    combined = (title + " " + article).str.strip()

    return pd.DataFrame({"text": combined, "label": label})


def main():
    fake = read_news(FAKE_PATH, label=0)
    real = read_news(REAL_PATH, label=1)

    data = pd.concat([fake, real], ignore_index=True)
    data["text"] = data["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    data = data[data["text"] != ""].drop_duplicates(subset="text")

    if data["label"].nunique() < 2:
        raise ValueError("The dataset must contain both fake and real examples.")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.20,
        random_state=42,
        stratify=data["label"],
    )

    model = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    stop_words="english",
                    max_features=50000,
                    ngram_range=(1, 2),
                    min_df=2,
                ),
            ),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"Test accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Fake", "Real"],
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
