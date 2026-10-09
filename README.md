# Fake News Detection System

A beginner-friendly final-year project that demonstrates text classification with **TF-IDF** and **Logistic Regression**, served through a **Flask** web application.

> **Important limitation:** This is an educational text-classification demo, not a fact-checking service. The model learns patterns from its training dataset and can misclassify genuine or misleading articles. Its confidence estimate is not proof of factual accuracy.

## Features

- Enter a news headline or article in a simple web interface.
- Predict whether the text is **Likely Real** or **Potentially Fake**.
- Display the model's estimated probability for its predicted class.
- Train the model locally and report test-set accuracy, precision, recall, and F1-score.
- Keep the dataset and generated model out of Git by default.

## Technology stack

- Python
- Flask
- pandas
- scikit-learn
- TF-IDF text vectorization
- Logistic Regression
- Joblib

## Project structure

```text
fake_news_detection/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── .gitignore
├── Fake.csv                 # Download separately; not committed
├── True.csv                 # Download separately; not committed
├── model/
│   └── fake_news_model.pkl  # Generated locally; not committed
└── templates/
    └── index.html
```

## Dataset setup

This project expects two CSV files named `Fake.csv` and `True.csv`, each containing `title` and `text` columns.

One commonly used educational dataset is the [Fake and Real News Dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).

1. Download the dataset from Kaggle and review its license and usage terms.
2. Extract the archive.
3. Copy `Fake.csv` and `True.csv` into the project root, beside `train_model.py`.
4. Do not upload the CSV files to GitHub unless you have confirmed that redistribution is permitted.

The dataset is historical and may contain source- or topic-specific patterns. It does not represent every news outlet, language, or current event.

## Setup and run

### 1. Install Python

Use a supported Python 3 version compatible with the listed packages.

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Train the model

Run this command from the project root:

```bash
python train_model.py
```

The script reads both CSV files, creates a stratified train/test split, trains the pipeline, prints evaluation metrics, and writes the generated model to `model/fake_news_model.pkl`.

### 5. Start the website

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

Stop the local server with `Ctrl+C`.

## How it works

1. **Input:** The user submits a headline or article.
2. **Text representation:** TF-IDF converts text into numeric features using word and two-word phrase patterns.
3. **Classification:** Logistic Regression predicts one of the two labels learned from the dataset.
4. **Output:** The web page displays the predicted label and the model's estimated probability for that prediction.

In the training data, label `0` represents the rows from `Fake.csv`, and label `1` represents the rows from `True.csv`.

## Evaluation and limitations

- The printed accuracy and classification report measure performance on a held-out portion of this dataset; they do not guarantee real-world accuracy.
- Dataset artifacts, duplicate stories, publisher-specific writing styles, and changing news topics can inflate or reduce measured performance.
- The displayed confidence is a model probability, not a calibrated probability that the claim is factually true.
- The system does not independently search the web, verify citations, or check claims against authoritative sources.
- Do not use this prototype for high-stakes decisions or to label a person or publisher as intentionally deceptive.

For a stronger research project, consider duplicate and near-duplicate leakage checks, time-based evaluation, external test data, calibration analysis, and comparisons with a simple baseline and other classifiers.

## Common issues

**`FileNotFoundError` for `Fake.csv` or `True.csv`**  
Ensure both files are in the same folder as `train_model.py`.

**`Model not found` when starting the app**  
Run `python train_model.py` successfully before starting `app.py`.

**CSV column error**  
Confirm that both CSV files contain columns named `title` and `text` (capitalization is not important).

**PowerShell will not activate the virtual environment**  
You can run the environment's Python executable directly, or consult the official Python documentation for your system's activation policy.

## Upload to GitHub

1. Create a new, empty repository on GitHub.
2. From the project folder, run:

```bash
git init
git add .
git commit -m "Add fake news detection project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

Replace the remote URL with your own repository URL. Because the dataset and generated model are ignored by `.gitignore`, Git will not upload them by default. A person cloning the repository must download the dataset, install dependencies, and train the model before running the app.

## Future improvements

- Add a claim-checking workflow using reliable external sources.
- Store prediction history in a database with appropriate privacy controls.
- Add automated tests and continuous integration.
- Compare multiple models and report per-class metrics.
- Add accessibility and mobile layout improvements.

## License

No license is specified for this project. Add a license file only after deciding how you want others to be allowed to use and redistribute your code. Dataset terms may be separate from your code's license.
