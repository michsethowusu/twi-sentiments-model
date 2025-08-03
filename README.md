# Twi Language Sentiment Classification Model

This repository contains a simple and fast **sentiment classification model** for detecting whether a sentence in **Twi (Akan)** has a Positive or Negative sentiment. It uses **FastText** for both training and inference and is designed to be easily adapted for other low-resource languages.

This repo can help you to:

- Verify if sentences in a Twi dataset have positive or negative sentiments, eg. reviews or tweets.
- Train a sentiment classification model for any other language using the same pipeline used for training this model.

---

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/michsethowusu/twi-id.git
cd twi-id
```

2. **Install dependencies**

```bash
pip install numpy==1.26.4 pandas==2.3.1 scikit-learn==1.7.1 fasttext==0.9.3
```


---

## 🚀 Usage

### 🔹 Predict Sentiment for a Single Sentence

```bash
python3 single-id.py
```

You'll be prompted to enter a sentence and the script will return whether it has a negative or positve sentiment.

---

### 🔹 Predict Sentiments for sentences in a CSV dataset

```bash
python3 batch-id.py
```

This script will:
1. Prompt you to select a CSV file (e.g. `sample_sentences.csv`)
2. Ask you to choose the column that contains the sentences
3. Output predictions to a new CSV

---

### 🔹 Train Your Own Model

To train on new data (e.g., for another language):

1. Prepare a `training-data.csv` with the columns and labels as per the sample data ie. column 1: sentence and column 2: sentiment.

2. Run the training script:
    ```bash
    python3 train.py
    ```

This will output a `twi_sentiment_model.bin` file inside the `model/` directory.

---

## 🧪 Sample Test File

You can test batch predictions with the sample provided:

```bash
data/sample_sentences.csv
```

---

## 🔒 License

MIT License — feel free to fork, adapt, and build upon it!

---

## 🙌 Acknowledgements

- Built with [FastText](https://fasttext.cc/)
- Part of the [GhanaNLP's](https://github.com/GhanaNLP) effort to make Ghanaian Languages accesible.

