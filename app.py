from flask import Flask, render_template, request
import joblib
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "fake_news_model.pkl"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Trained model not found. Run `python train_model.py` first."
        )
    return joblib.load(MODEL_PATH)


model = load_model()


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = None
    news_text = ""
    error = None

    if request.method == "POST":
        news_text = request.form.get("news_text", "").strip()

        if not news_text:
            error = "Please enter a news headline or article."
        else:
            prediction = int(model.predict([news_text])[0])
            probabilities = model.predict_proba([news_text])[0]
            confidence = round(float(max(probabilities)) * 100, 2)
            result = "Likely Real" if prediction == 1 else "Potentially Fake"

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        news_text=news_text,
        error=error,
    )


if __name__ == "__main__":
    # Debug is disabled by default for safer local use.
    app.run(debug=False)
