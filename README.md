# Twi Language Sentiment Classification Model

This repository contains a model for classigifcation of sentneces in Twi language into negative or positive sentiments. It also contains dataset uses for the training and code for training a fasttext based **sentiment classification model** for **Twi (Akan)** or any other low-resource languages. You can download the model and dataset from the releases - https://github.com/michsethowusu/twi-senti/releases.

---

## Setup

1. **Clone the repository**

```bash
git clone https://github.com/michsethowusu/twi-senti.git
cd twi-senti
```

2. **Install dependencies**

```bash
pip install numpy==1.26.4 pandas==2.3.1 scikit-learn==1.7.1 fasttext==0.9.3
```


---

## Usage

### 🔹 Train Your Own Model

To train on new data (e.g., for another language):

1. Prepare a `training-data.csv` with the columns and labels as per the sample data ie. column 1: sentence and column 2: sentiment.

2. Run the training script:
    ```bash
    python3 train.py
    ```

This will output a `twi_sentiment_model.bin` file inside the `model/` directory.

---

### Predict Sentiment for a Single Sentence

```bash
python3 predict_single.py
```

You'll be prompted to enter a sentence and the script will return whether it has a negative or positve sentiment.

---

### Predict Sentiments for sentences in a CSV dataset

```bash
python3 predict_batch.py
```

This script will:
1. Prompt you to select a CSV file (e.g. `sample_sentences.csv`)
2. Ask you to choose the column that contains the sentences
3. Output predictions to a new CSV

---

## License

MIT License — feel free to fork, adapt, and build upon it!

---

## Acknowledgements

- Built with [FastText](https://fasttext.cc/)
- Part of the [GhanaNLP's](https://github.com/GhanaNLP) effort to make Ghanaian Languages accesible.

