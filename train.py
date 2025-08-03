# Step 1: Import libraries
import pandas as pd
import fasttext
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Step 2: Specify your CSV path
csv_path = "data/training-dataset.csv"  # CHANGE THIS TO YOUR CSV PATH
print(f"Loading CSV from: {csv_path}")

# Step 3: Read CSV file
try:
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded successfully with {len(df):,} samples")
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}")
    print("Please update the csv_path variable with the correct path to your CSV file.")
    exit()
except Exception as e:
    print(f"Error reading CSV: {e}")
    exit()

# Step 4: Explore the dataset
print("\nDataset info:")
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")

# Check if the expected columns exist
expected_columns = ['sentence', 'sentiment']
missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    print(f"\nWarning: Missing expected columns: {missing_columns}")
    print("Available columns:", df.columns.tolist())
    print("\nPlease ensure your CSV has columns named 'sentence' and 'sentiment'")
    print("Or modify the column names below:")
    
    # Auto-detect likely column names
    text_col = None
    sentiment_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'sentence' in col_lower or 'text' in col_lower or 'twi' in col_lower:
            text_col = col
        elif 'sentiment' in col_lower or 'label' in col_lower or 'emotion' in col_lower:
            sentiment_col = col
    
    if text_col and sentiment_col:
        print(f"\nAuto-detected columns:")
        print(f"Text column: '{text_col}'")
        print(f"Sentiment column: '{sentiment_col}'")
        
        # Rename columns for consistency
        df = df.rename(columns={text_col: 'sentence', sentiment_col: 'sentiment'})
        print("Columns renamed for processing.")
    else:
        print("\nCould not auto-detect columns. Please check your CSV format.")
        exit()

print("\nSentiment distribution:")
sentiment_counts = df['sentiment'].value_counts()
print(sentiment_counts)

# Show percentage distribution
for sentiment, count in sentiment_counts.items():
    percentage = count / len(df) * 100
    print(f"{sentiment}: {count:,} ({percentage:.1f}%)")

print("\nSample data:")
for i, row in df.head(10).iterrows():
    text_preview = row['sentence'][:80] + '...' if len(str(row['sentence'])) > 80 else str(row['sentence'])
    print(f"{i+1}. [{row['sentiment']}] {text_preview}")

# Step 5: Prepare data for FastText
print("\nPreparing data...")

# Clean the text data
df['sentence'] = df['sentence'].astype(str).str.strip()
df['sentiment'] = df['sentiment'].astype(str).str.strip()

# Remove any empty or null entries
original_len = len(df)
df = df.dropna(subset=['sentence', 'sentiment'])
df = df[df['sentence'].str.len() > 0]
df = df[df['sentiment'].str.len() > 0]

if len(df) < original_len:
    print(f"Removed {original_len - len(df)} empty/null entries")

# Check unique sentiment labels
unique_sentiments = df['sentiment'].unique()
print(f"\nUnique sentiment labels found: {unique_sentiments}")

# For binary sentiment classification, ensure we have Positive/Negative
# If your data has different labels, you can map them here
label_mapping = {
    # Add mappings if needed, e.g.:
    # 'pos': 'Positive',
    # 'neg': 'Negative',
    # '1': 'Positive',
    # '0': 'Negative',
    # 'positive': 'Positive',
    # 'negative': 'Negative'
}

if label_mapping:
    df['sentiment'] = df['sentiment'].map(label_mapping).fillna(df['sentiment'])
    print(f"Applied label mapping: {label_mapping}")

# Filter to keep only valid sentiment labels
valid_sentiments = ['Positive', 'Negative']
before_filter = len(df)
df = df[df['sentiment'].isin(valid_sentiments)]
after_filter = len(df)

if before_filter != after_filter:
    print(f"Filtered to valid sentiments: {before_filter} -> {after_filter} samples")
    removed_labels = set(unique_sentiments) - set(valid_sentiments)
    if removed_labels:
        print(f"Removed samples with labels: {removed_labels}")

if len(df) == 0:
    print("Error: No valid sentiment data remaining after filtering.")
    print(f"Expected labels: {valid_sentiments}")
    print(f"Found labels: {unique_sentiments}")
    exit()

print(f"\nFinal dataset: {len(df):,} samples")
final_counts = df['sentiment'].value_counts()
print("Final sentiment distribution:")
for sentiment, count in final_counts.items():
    percentage = count / len(df) * 100
    print(f"  {sentiment}: {count:,} ({percentage:.1f}%)")

# Convert to FastText format
df['fasttext_format'] = '__label__' + df['sentiment'] + ' ' + df['sentence'].str.replace('\n', ' ').str.replace('\r', ' ')

# Step 6: Split data (80% train, 20% test)
print("\nSplitting data...")
train_data, test_data = train_test_split(
    df['fasttext_format'], 
    test_size=0.2, 
    random_state=42,
    stratify=df['sentiment']  # Ensure balanced split across sentiments
)

# Save datasets
train_file = "sentiment_train.txt"
test_file = "sentiment_test.txt"

# Save as text files (FastText format)
with open(train_file, 'w', encoding='utf-8') as f:
    for line in train_data:
        f.write(line + '\n')

with open(test_file, 'w', encoding='utf-8') as f:
    for line in test_data:
        f.write(line + '\n')

print(f"Dataset splits saved:")
print(f"Train: {len(train_data):,} samples -> {train_file}")
print(f"Test: {len(test_data):,} samples -> {test_file}")

# Step 7: Train FastText sentiment classification model
print("\nTraining FastText sentiment classification model...")
print("This may take a few minutes depending on dataset size...")

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
    maxn=6               # Max char n-gram length (helps with OOV words)
)

print("Model training completed!")

# Step 8: Evaluate the model
print("\nEvaluating model on test set...")
results = model.test(test_file)
print(f"Precision: {results[1]:.4f}")
print(f"Recall: {results[2]:.4f}")
print(f"Number of test examples: {results[0]:,}")

# Step 9: Save the model
model_path = "model/twi_sentiment_model.bin"
model.save_model(model_path)
print(f"\nModel saved as '{model_path}'")

# Step 10: Safe prediction function for sentiment classification
def safe_predict_sentiment(model, text, k=2):
    """
    Safe prediction function for sentiment classification
    """
    try:
        labels, probabilities = model.predict(text, k=k)
        return [(label.replace('__label__', ''), prob) for label, prob in zip(labels, probabilities)]
    except ValueError:
        # Fallback for NumPy compatibility issues
        try:
            prediction = model.predict(text)
            if isinstance(prediction, tuple) and len(prediction) >= 2:
                labels = prediction[0]
                try:
                    probabilities = prediction[1]
                    return [(label.replace('__label__', ''), prob) for label, prob in zip(labels[:k], probabilities[:k])]
                except:
                    return [(label.replace('__label__', ''), 0.0) for label in labels[:k]]
            else:
                return [("unknown", 0.0)]
        except Exception as e:
            print(f"Prediction error: {e}")
            return [("error", 0.0)]

# Step 11: Test predictions with sample sentences
print("\n" + "="*60)
print("TESTING SENTIMENT PREDICTIONS")
print("="*60)

# Use actual samples from your dataset for testing
sample_data = df.sample(n=min(5, len(df)), random_state=42)
print("Testing with samples from your dataset:")

for _, row in sample_data.iterrows():
    text = row['sentence']
    true_sentiment = row['sentiment']
    
    print(f"\nSentence: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    print(f"True sentiment: {true_sentiment}")
    
    predictions = safe_predict_sentiment(model, text, k=2)
    print("Predicted sentiments:")
    for i, (sentiment, confidence) in enumerate(predictions):
        marker = "✓" if sentiment == true_sentiment else " "
        print(f"  {marker} {i+1}. {sentiment}: {confidence:.4f}")

# Additional test sentences (you can modify these)
additional_tests = [
    "Me ani agye yiye nnɛ",      # "I am very happy today"
    "Me werɛ ahow yiye",         # "I am very sad"
    "Ɛyɛ amane paa",             # "It's very bad"
    "Ɛyɛ fɛ yiye",               # "It's very beautiful"
    "Medɔ wo",                   # "I love you"
]

print(f"\nTesting additional Twi sentences:")
for sentence in additional_tests:
    print(f"\nSentence: '{sentence}'")
    predictions = safe_predict_sentiment(model, sentence, k=2)
    print("Sentiment predictions:")
    for i, (sentiment, confidence) in enumerate(predictions):
        print(f"  {i+1}. {sentiment}: {confidence:.4f}")

# Step 12: Detailed evaluation on sample data
print("\n" + "="*60)
print("DETAILED EVALUATION")
print("="*60)

# Evaluate on a larger sample
eval_sample_size = min(1000, len(df))
test_df = df.sample(n=eval_sample_size, random_state=42)

y_true = []
y_pred = []

print(f"Analyzing predictions on {eval_sample_size} samples...")
for _, row in test_df.iterrows():
    true_sentiment = row['sentiment']
    text = row['sentence']
    
    predictions = safe_predict_sentiment(model, text, k=1)
    predicted_sentiment = predictions[0][0] if predictions else "unknown"
    
    y_true.append(true_sentiment)
    y_pred.append(predicted_sentiment)

# Calculate accuracy
correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
accuracy = correct / len(y_true)

print(f"\nEvaluation results on {eval_sample_size} samples:")
print(f"Accuracy: {accuracy:.4f} ({correct}/{len(y_true)})")

# Count predictions by sentiment
from collections import Counter
true_counts = Counter(y_true)
pred_counts = Counter(y_pred)

print(f"\nTrue sentiment distribution in sample:")
for sentiment, count in true_counts.items():
    print(f"  {sentiment}: {count} ({count/len(y_true)*100:.1f}%)")

print(f"\nPredicted sentiment distribution in sample:")
for sentiment, count in pred_counts.items():
    print(f"  {sentiment}: {count} ({count/len(y_pred)*100:.1f}%)")

# Per-sentiment accuracy
print(f"\nPer-sentiment accuracy:")
for sentiment in ['Positive', 'Negative']:
    if sentiment in true_counts:
        sentiment_indices = [i for i, s in enumerate(y_true) if s == sentiment]
        sentiment_correct = sum(1 for i in sentiment_indices if y_true[i] == y_pred[i])
        sentiment_total = len(sentiment_indices)
        sentiment_acc = sentiment_correct / sentiment_total if sentiment_total > 0 else 0
        print(f"  {sentiment}: {sentiment_acc:.4f} ({sentiment_correct}/{sentiment_total})")

# Step 13: Interactive testing function
def test_sentiment(text):
    """
    Quick function to test sentiment of any Twi text
    """
    predictions = safe_predict_sentiment(model, text, k=2)
    print(f"\nText: '{text}'")
    print("Sentiment Analysis:")
    for sentiment, confidence in predictions:
        print(f"  {sentiment}: {confidence:.4f}")
    
    return predictions[0][0] if predictions else "unknown"

print("\n" + "="*60)
print("MODEL TRAINING COMPLETE!")
print(f"Model saved as: {model_path}")
print("You can now use this model to classify sentiment in Twi text.")
print(f"Training completed with {len(df):,} samples")
print(f"Final accuracy on sample: {accuracy:.4f}")
print("\nUsage:")
print("1. Update csv_path variable at the top of the script")
print("2. Ensure your CSV has 'sentence' and 'sentiment' columns")
print("3. Run the script to train your model")
print("4. Use test_sentiment('Your text here') to test predictions")
print("="*60)
