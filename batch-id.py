# src/batch_predict_interactive.py
import fasttext
import pandas as pd
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import os
from datetime import datetime

def safe_predict_sentiment(model, text, k=2):
    """
    Safe prediction function for sentiment classification that handles NumPy/FastText quirks
    """
    try:
        labels, probs = model.predict(text, k=k)
        return [(label.replace('__label__', ''), prob) for label, prob in zip(labels, probs)]
    except Exception as e:
        print(f"Prediction error for text '{text[:50]}...': {e}")
        return [("error", 0.0)]

def choose_column(columns):
    print("\n📋 Available columns in the CSV:")
    for i, col in enumerate(columns):
        print(f"  {i + 1}. {col}")
    
    while True:
        try:
            choice = int(input("\n🔢 Enter the number of the column that contains the sentences: "))
            if 1 <= choice <= len(columns):
                return columns[choice - 1]
            else:
                print("❗ Invalid selection. Try again.")
        except ValueError:
            print("❗ Please enter a number.")

def get_sentiment_summary(df):
    """
    Generate a summary of sentiment predictions
    """
    sentiment_counts = df['predicted_sentiment'].value_counts()
    total = len(df)
    
    print(f"\n📊 SENTIMENT ANALYSIS SUMMARY")
    print(f"{'='*40}")
    print(f"Total sentences analyzed: {total:,}")
    
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total) * 100
        print(f"  {sentiment}: {count:,} ({percentage:.1f}%)")
    
    # Average confidence
    avg_confidence = df['confidence'].mean()
    print(f"\nAverage confidence: {avg_confidence:.3f}")
    
    # High confidence predictions (> 0.8)
    high_conf = df[df['confidence'] > 0.8]
    high_conf_pct = (len(high_conf) / total) * 100
    print(f"High confidence predictions (>0.8): {len(high_conf):,} ({high_conf_pct:.1f}%)")
    
    return sentiment_counts

def main():
    print("🎭 Twi Sentiment Batch Analyzer")
    print("="*40)
    
    # Initialize Tkinter root and hide the main window
    root = tk.Tk()
    root.withdraw()
    
    # Prompt for CSV input file using file dialog
    print("\n📥 Please select the input CSV file...")
    input_path = filedialog.askopenfilename(
        title="Select Input CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    if not input_path:
        print("❌ No file selected. Exiting.")
        return
    
    input_file = Path(input_path)
    print(f"📁 Selected file: {input_file.name}")
    
    # Load CSV
    try:
        df = pd.read_csv(input_file)
        print(f"✅ Loaded CSV with {len(df):,} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return
    
    # Show first few rows as preview
    print(f"\n👀 Preview of data:")
    print(df.head(3).to_string(max_cols=5, max_colwidth=50))
    
    # Ask user to pick column
    text_column = choose_column(df.columns.tolist())
    print(f"✅ Selected column: '{text_column}'")
    
    # Check for empty/null values in selected column
    null_count = df[text_column].isnull().sum()
    empty_count = (df[text_column].astype(str).str.strip() == '').sum()
    
    if null_count > 0 or empty_count > 0:
        print(f"⚠️  Found {null_count} null and {empty_count} empty values in column '{text_column}'")
        clean_choice = input("Remove empty rows? (y/n): ").strip().lower()
        if clean_choice == 'y':
            original_len = len(df)
            df = df.dropna(subset=[text_column])
            df = df[df[text_column].astype(str).str.strip() != '']
            print(f"✅ Cleaned data: {original_len} → {len(df)} rows")
    
    # Create data folder if it doesn't exist
    data_folder = Path("data")
    data_folder.mkdir(exist_ok=True)
    
    # Generate output filename based on input filename
    input_filename = input_file.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{input_filename}_sentiment_predictions_{timestamp}.csv"
    output_file = data_folder / output_filename
    
    # Check if output file exists and prompt for overwrite
    if output_file.exists():
        overwrite = input(f"\n⚠️ File {output_file} already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("❌ Operation cancelled.")
            return
    
    # Load sentiment model
    model_path = Path("model/twi_sentiment_model.bin")  # Updated path
    if not model_path.exists():
        print(f"❌ Sentiment model not found at '{model_path}'.")
        print("   Please ensure 'twi_sentiment_model.bin' exists in the current directory.")
        return
    
    print(f"\n🔄 Loading sentiment model from {model_path}...")
    try:
        model = fasttext.load_model(str(model_path))
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"\n🔎 Running sentiment predictions on {len(df):,} rows from column '{text_column}'...")
    print("This may take a moment for large files...")
    
    # Process predictions with progress indication
    predictions = []
    total_rows = len(df)
    
    for i, text in enumerate(df[text_column].astype(str)):
        if i % 1000 == 0 and i > 0:  # Progress update every 1000 rows
            print(f"   Processed {i:,}/{total_rows:,} rows ({(i/total_rows)*100:.1f}%)")
        
        pred = safe_predict_sentiment(model, text, k=2)
        predictions.append(pred)
    
    print(f"✅ Completed predictions for all {total_rows:,} rows!")
    
    # Extract top predictions and all predictions
    df['predicted_sentiment'] = [pred[0][0] if pred else "error" for pred in predictions]
    df['confidence'] = [pred[0][1] if pred else 0.0 for pred in predictions]
    
    # Add secondary prediction for binary classification
    df['secondary_sentiment'] = [pred[1][0] if len(pred) > 1 else "" for pred in predictions]
    df['secondary_confidence'] = [pred[1][1] if len(pred) > 1 else 0.0 for pred in predictions]
    
    # Add confidence category
    def categorize_confidence(conf):
        if conf > 0.8:
            return "High"
        elif conf > 0.6:
            return "Medium"
        elif conf > 0.4:
            return "Low"
        else:
            return "Very Low"
    
    df['confidence_category'] = df['confidence'].apply(categorize_confidence)
    
    # Save results
    try:
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return
    
    # Generate and display summary
    sentiment_counts = get_sentiment_summary(df)
    
    # Show some example predictions
    print(f"\n🔍 SAMPLE PREDICTIONS:")
    print(f"{'='*60}")
    
    sample_df = df.sample(n=min(5, len(df)), random_state=42)
    for _, row in sample_df.iterrows():
        text = str(row[text_column])
        text_preview = text[:80] + "..." if len(text) > 80 else text
        sentiment = row['predicted_sentiment']
        confidence = row['confidence']
        
        emoji = "😊" if sentiment == "Positive" else "😞" if sentiment == "Negative" else "❓"
        print(f"\nText: \"{text_preview}\"")
        print(f"Sentiment: {emoji} {sentiment} (confidence: {confidence:.3f})")
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {output_file}")
    print(f"📊 Total sentences: {len(df):,}")
    print(f"🎯 Average confidence: {df['confidence'].mean():.3f}")

if __name__ == "__main__":
    main()
