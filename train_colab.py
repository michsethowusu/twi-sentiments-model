# =====================================================================
# TWI SENTIMENT ANALYSIS FOR GOOGLE COLAB
# =====================================================================

# CONFIGURATION - CHANGE THESE PATHS AS NEEDED
DATASET_PATH = "/content/drive/MyDrive/Collab/training-dataset.csv"  # 📁 Change this to your dataset path
MODEL_NAME = "twi_sentiment_model"              # 🏷️ Name for your saved model
OUTPUT_DIR = "/content/model"                   # 📂 Directory to save model and files

# =====================================================================
# SETUP AND INSTALLATION
# =====================================================================

# Import libraries
import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from pathlib import Path
from collections import Counter

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("TWI SENTIMENT ANALYSIS MODEL TRAINER")
print("="*60)
print(f"📁 Dataset Path: {DATASET_PATH}")
print(f"🏷️ Model Name: {MODEL_NAME}")
print(f"📂 Output Directory: {OUTPUT_DIR}")
print("="*60)

# =====================================================================
# LOAD AND EXPLORE DATASET
# =====================================================================

print(f"Loading dataset from: {DATASET_PATH}")

# Check if file exists and provide helpful messages
if not os.path.exists(DATASET_PATH):
    print(f"❌ Error: File not found at {DATASET_PATH}")
    print("\n🔍 Looking for CSV files in /content/...")

    # Search for CSV files in /content/
    csv_files = []
    for root, dirs, files in os.walk("/content/"):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))

    if csv_files:
        print(f"Found {len(csv_files)} CSV file(s):")
        for i, file in enumerate(csv_files[:10], 1):  # Show first 10
            print(f"  {i}. {file}")
        print(f"\n💡 Update DATASET_PATH variable above to use one of these files")
    else:
        print("No CSV files found in /content/")
        print("\n📤 Upload your CSV file to Colab:")
        print("1. Click the folder icon on the left sidebar")
        print("2. Upload your CSV file")
        print("3. Update DATASET_PATH with the uploaded file path")

    raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")

try:
    # Load the dataset
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Dataset loaded successfully with {len(df):,} samples")

    # Display basic info
    print(f"\n📊 Dataset Info:")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")

    # Show first few rows
    print(f"\n👀 First 3 rows:")
    print(df.head(3))

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    raise

# =====================================================================
# DATA PREPARATION AND CLEANING
# =====================================================================

print(f"\n🔧 Preparing data...")

# Check if the expected columns exist
expected_columns = ['sentence', 'sentiment']
missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    print(f"⚠️ Missing expected columns: {missing_columns}")
    print(f"Available columns: {df.columns.tolist()}")

    # Auto-detect likely column names
    text_col = None
    sentiment_col = None

    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['sentence', 'text', 'twi', 'content', 'message', 'review']):
            text_col = col
        elif any(keyword in col_lower for keyword in ['sentiment', 'label', 'emotion', 'class', 'target']):
            sentiment_col = col

    if text_col and sentiment_col:
        print(f"🔍 Auto-detected columns:")
        print(f"   Text column: '{text_col}'")
        print(f"   Sentiment column: '{sentiment_col}'")

        # Rename columns for consistency
        df = df.rename(columns={text_col: 'sentence', sentiment_col: 'sentiment'})
        print("✅ Columns renamed for processing")
    else:
        print("❌ Could not auto-detect columns")
        print("Please ensure your CSV has appropriate column names or modify the code")
        raise ValueError("Column detection failed")

# Clean the data
print(f"\n🧹 Cleaning data...")
original_len = len(df)

# Convert to string and clean
df['sentence'] = df['sentence'].astype(str).str.strip()
df['sentiment'] = df['sentiment'].astype(str).str.strip()

# Remove empty/null entries
df = df.dropna(subset=['sentence', 'sentiment'])
df = df[df['sentence'].str.len() > 0]
df = df[df['sentiment'].str.len() > 0]

if len(df) < original_len:
    print(f"🗑️ Removed {original_len - len(df)} empty/null entries")

# Check sentiment labels
unique_sentiments = df['sentiment'].unique()
print(f"\n🏷️ Unique sentiment labels: {unique_sentiments}")

# Show distribution
sentiment_counts = df['sentiment'].value_counts()
print(f"\n📈 Sentiment distribution:")
for sentiment, count in sentiment_counts.items():
    percentage = count / len(df) * 100
    print(f"   {sentiment}: {count:,} ({percentage:.1f}%)")

# =====================================================================
# LABEL STANDARDIZATION
# =====================================================================

print(f"\n🔄 Standardizing labels...")

# Map various label formats to standard Positive/Negative
label_mapping = {
    'pos': 'Positive', 'positive': 'Positive', 'good': 'Positive', '1': 'Positive',
    'neg': 'Negative', 'negative': 'Negative', 'bad': 'Negative', '0': 'Negative'
}

# Apply case-insensitive mapping
for old_label, new_label in label_mapping.items():
    mask = df['sentiment'].str.lower() == old_label.lower()
    if mask.any():
        df.loc[mask, 'sentiment'] = new_label
        print(f"   '{old_label}' → '{new_label}' ({mask.sum()} samples)")

# Filter to keep only valid sentiment labels
valid_sentiments = ['Positive', 'Negative']
before_filter = len(df)
df = df[df['sentiment'].isin(valid_sentiments)]
after_filter = len(df)

if before_filter != after_filter:
    print(f"🔍 Filtered to valid sentiments: {before_filter:,} → {after_filter:,} samples")

if len(df) == 0:
    print("❌ No valid sentiment data remaining after filtering")
    print(f"Expected labels: {valid_sentiments}")
    raise ValueError("No valid data after filtering")

# Final statistics
print(f"\n✅ Final dataset: {len(df):,} samples")
final_counts = df['sentiment'].value_counts()
for sentiment, count in final_counts.items():
    percentage = count / len(df) * 100
    print(f"   {sentiment}: {count:,} ({percentage:.1f}%)")

# Show sample data
print(f"\n📝 Sample data:")
for i, row in df.head(5).iterrows():
    text_preview = row['sentence'][:60] + '...' if len(str(row['sentence'])) > 60 else str(row['sentence'])
    print(f"   {i+1}. [{row['sentiment']}] {text_preview}")

# =====================================================================
# PREPARE FASTTEXT FORMAT AND SPLIT DATA
# =====================================================================

print(f"\n✂️ Preparing data splits...")

# Convert to FastText format
df['fasttext_format'] = '__label__' + df['sentiment'] + ' ' + df['sentence'].str.replace('\n', ' ').str.replace('\r', ' ')

# Split data (80% train, 20% test)
train_data, test_data = train_test_split(
    df['fasttext_format'],
    test_size=0.2,
    random_state=42,
    stratify=df['sentiment']
)

# Save datasets
train_file = f"{OUTPUT_DIR}/sentiment_train.txt"
test_file = f"{OUTPUT_DIR}/sentiment_test.txt"

with open(train_file, 'w', encoding='utf-8') as f:
    for line in train_data:
        f.write(line + '\n')

with open(test_file, 'w', encoding='utf-8') as f:
    for line in test_data:
        f.write(line + '\n')

print(f"💾 Dataset splits saved:")
print(f"   Train: {len(train_data):,} samples → {train_file}")
print(f"   Test: {len(test_data):,} samples → {test_file}")

# =====================================================================
# TRAIN FASTTEXT MODEL
# =====================================================================

print(f"\n🚀 Training FastText model...")
print("⏱️ This may take a few minutes depending on dataset size...")

try:
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.1,              # Learning rate
        epoch=25,            # Number of epochs
        wordNgrams=2,        # N-grams for context
        dim=150,             # Dimension of word vectors
        loss='ova',          # One-vs-all for binary classification
        thread=8,
        minCount=3,          # Minimum word count
        ws=5,                # Window size for context
        minn=3,              # Min char n-gram length
        maxn=6               # Max char n-gram length
    )

    print("✅ Model training completed!")

except Exception as e:
    print(f"❌ Training failed: {e}")
    raise

# =====================================================================
# EVALUATE MODEL
# =====================================================================

print(f"\n📊 Evaluating model...")

# Test on test set
results = model.test(test_file)
precision = results[1]
recall = results[2]
n_samples = results[0]

print(f"📈 Test Results:")
print(f"   Samples: {n_samples:,}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall: {recall:.4f}")

# Save the model
model_path = f"{OUTPUT_DIR}/{MODEL_NAME}.bin"
model.save_model(model_path)
print(f"\n💾 Model saved: {model_path}")

# =====================================================================
# PREDICTION FUNCTION
# =====================================================================

def predict_sentiment(text, k=2):
    """
    Predict sentiment for given text

    Args:
        text (str): Input text to analyze
        k (int): Number of top predictions to return

    Returns:
        list: List of (sentiment, confidence) tuples
    """
    try:
        labels, probabilities = model.predict(text, k=k)
        return [(label.replace('__label__', ''), prob) for label, prob in zip(labels, probabilities)]
    except Exception as e:
        print(f"Prediction error: {e}")
        return [("error", 0.0)]

# =====================================================================
# TEST PREDICTIONS
# =====================================================================

print(f"\n🧪 Testing predictions...")
print("="*50)

# Test with actual samples from dataset
print("📋 Testing with dataset samples:")
sample_data = df.sample(n=min(5, len(df)), random_state=42)

for idx, (_, row) in enumerate(sample_data.iterrows(), 1):
    text = row['sentence']
    true_sentiment = row['sentiment']

    print(f"\n{idx}. Text: '{text[:80]}{'...' if len(text) > 80 else ''}'")
    print(f"   True: {true_sentiment}")

    predictions = predict_sentiment(text, k=2)
    print(f"   Predicted:")
    for i, (sentiment, confidence) in enumerate(predictions):
        marker = "✅" if sentiment == true_sentiment else "❌"
        print(f"      {marker} {sentiment}: {confidence:.4f}")

# Test with additional Twi sentences
print(f"\n🗣️ Testing with sample Twi sentences:")
test_sentences = [
    "Me ani agye yiye nnɛ",      # "I am very happy today"
    "Me werɛ ahow yiye",         # "I am very sad"
    "Ɛyɛ amane paa",             # "It's very bad"
    "Ɛyɛ fɛ yiye",               # "It's very beautiful"
    "Medɔ wo",                   # "I love you"
]

for idx, sentence in enumerate(test_sentences, 1):
    print(f"\n{idx}. '{sentence}'")
    predictions = predict_sentiment(sentence, k=2)
    for sentiment, confidence in predictions:
        print(f"   {sentiment}: {confidence:.4f}")

# =====================================================================
# DETAILED EVALUATION
# =====================================================================

print(f"\n📊 Detailed evaluation...")
print("="*50)

# Evaluate on larger sample
eval_size = min(1000, len(df))
eval_df = df.sample(n=eval_size, random_state=42)

y_true = []
y_pred = []

for _, row in eval_df.iterrows():
    true_sentiment = row['sentiment']
    text = row['sentence']

    predictions = predict_sentiment(text, k=1)
    predicted_sentiment = predictions[0][0] if predictions else "unknown"

    y_true.append(true_sentiment)
    y_pred.append(predicted_sentiment)

# Calculate metrics
correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
accuracy = correct / len(y_true)

print(f"📈 Evaluation on {eval_size:,} samples:")
print(f"   Overall Accuracy: {accuracy:.4f} ({correct}/{len(y_true)})")

# Per-sentiment accuracy
true_counts = Counter(y_true)
print(f"\n📊 Per-sentiment accuracy:")
for sentiment in ['Positive', 'Negative']:
    if sentiment in true_counts:
        sentiment_indices = [i for i, s in enumerate(y_true) if s == sentiment]
        sentiment_correct = sum(1 for i in sentiment_indices if y_true[i] == y_pred[i])
        sentiment_total = len(sentiment_indices)
        sentiment_acc = sentiment_correct / sentiment_total if sentiment_total > 0 else 0
        print(f"   {sentiment}: {sentiment_acc:.4f} ({sentiment_correct}/{sentiment_total})")

# =====================================================================
# COMPLETION SUMMARY
# =====================================================================

print(f"\n" + "="*60)
print("🎉 MODEL TRAINING COMPLETED!")
print("="*60)
print(f"📁 Dataset: {DATASET_PATH}")
print(f"📊 Samples: {len(df):,}")
print(f"🎯 Accuracy: {accuracy:.4f}")
print(f"💾 Model: {model_path}")
print(f"📂 Files: {OUTPUT_DIR}/")
print("="*60)

print(f"\n🚀 Quick Usage:")
print(f"   result = predict_sentiment('Your Twi text here')")
print(f"   print(result)")

print(f"\n📝 Example:")
example_text = "Me ani agye yiye"
example_result = predict_sentiment(example_text)
print(f"   predict_sentiment('{example_text}')")
print(f"   → {example_result}")

print(f"\n✨ Your model is ready to use! ✨")
