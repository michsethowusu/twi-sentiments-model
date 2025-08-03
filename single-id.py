# src/predict.py
import fasttext
import sys
from pathlib import Path

def safe_predict_sentiment(model, text, k=2):
    """
    Safe prediction function for sentiment classification that handles NumPy compatibility issues
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

def get_sentiment_emoji(sentiment, confidence):
    """
    Return appropriate emoji based on sentiment and confidence
    """
    if sentiment.lower() == 'positive':
        if confidence > 0.8:
            return "😊"  # Very confident positive
        elif confidence > 0.6:
            return "🙂"  # Moderately confident positive
        else:
            return "😐"  # Low confidence positive
    elif sentiment.lower() == 'negative':
        if confidence > 0.8:
            return "😞"  # Very confident negative
        elif confidence > 0.6:
            return "😕"  # Moderately confident negative
        else:
            return "😐"  # Low confidence negative
    else:
        return "❓"  # Unknown sentiment

def format_confidence_bar(confidence, width=20):
    """
    Create a visual confidence bar
    """
    filled = int(confidence * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"

def main():
    # Set model path - updated for sentiment model
    model_path = Path("model/twi_sentiment_model.bin")  # Updated path
    
    if not model_path.exists():
        print("❌ Sentiment model file not found!")
        print(f"   Looking for: {model_path}")
        print("   Please ensure 'twi_sentiment_model.bin' exists in the current directory.")
        print("   Or update the model_path variable to point to your model file.")
        return

    try:
        # Load sentiment model
        print("🔄 Loading sentiment model...")
        model = fasttext.load_model(str(model_path))
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # Get input text
    if len(sys.argv) > 1:
        # Use command-line argument if provided
        input_text = " ".join(sys.argv[1:])
    else:
        # Interactive mode
        print("\n" + "="*50)
        print("🎭 TWI SENTIMENT ANALYZER")
        print("="*50)
        print("Enter Twi text to analyze sentiment (or 'quit' to exit)")
        
        while True:
            input_text = input("\n📝 Enter text: ").strip()
            
            if not input_text:
                print("⚠️  No input provided. Please enter some text.")
                continue
            
            if input_text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                return
            
            # Analyze sentiment
            analyze_sentiment(model, input_text)

def analyze_sentiment(model, input_text):
    """
    Analyze sentiment of the input text and display results
    """
    # Get predictions (top 2 for binary classification)
    predictions = safe_predict_sentiment(model, input_text, k=2)
    
    if not predictions or predictions[0][0] == "error":
        print("❌ Error analyzing sentiment. Please try again.")
        return
    
    # Get top prediction
    top_sentiment, top_confidence = predictions[0]
    
    # Get emoji for visualization
    emoji = get_sentiment_emoji(top_sentiment, top_confidence)
    
    # Confidence level description
    if top_confidence > 0.8:
        confidence_desc = "Very High"
    elif top_confidence > 0.6:
        confidence_desc = "High"
    elif top_confidence > 0.4:
        confidence_desc = "Moderate"
    else:
        confidence_desc = "Low"
    
    print(f"\n🔍 Analysis Results:")
    print(f"   Text: \"{input_text}\"")
    print(f"   Sentiment: {emoji} {top_sentiment.upper()}")
    print(f"   Confidence: {top_confidence:.3f} ({confidence_desc})")
    print(f"   Visual: {format_confidence_bar(top_confidence)} {top_confidence:.1%}")
    
    # Show all predictions if more than one
    if len(predictions) > 1:
        print(f"\n📊 All Predictions:")
        for i, (sentiment, confidence) in enumerate(predictions):
            emoji = get_sentiment_emoji(sentiment, confidence)
            print(f"   {i+1}. {emoji} {sentiment}: {confidence:.3f} ({confidence:.1%})")

if __name__ == "__main__":
    # Check if running in command-line mode
    if len(sys.argv) > 1:
        # Command-line mode
        model_path = Path("model/twi_sentiment_model.bin")
        
        if not model_path.exists():
            print("❌ Sentiment model file not found!")
            print(f"   Looking for: {model_path}")
            sys.exit(1)
        
        try:
            model = fasttext.load_model(str(model_path))
            input_text = " ".join(sys.argv[1:])
            analyze_sentiment(model, input_text)
        except Exception as e:
            print(f"❌ Error: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        main()
