import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


def load_model_and_tokenizer(model_dir: str):
    """
    Load a saved model directory, including tokenizer and id2label mapping.
    """
    print(f"Loading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Normalize id2label keys to int
    raw_id2label = model.config.id2label
    id2label = {int(k): v for k, v in raw_id2label.items()}

    return tokenizer, model, id2label


def predict_event(text: str, tokenizer, model, id2label, threshold=0.5, device=None):
    """
    Predict event types for a given text using multi-label classification.
    
    Args:
        text: Input text to classify
        tokenizer: Tokenizer instance
        model: Model instance
        id2label: Dict mapping label indices to names
        threshold: Probability threshold (float or dict/array of thresholds per class)
        device: Device to use (cuda, mps, cpu). If None, auto-detects.
    
    Returns:
        Dict with prediction results including all predicted labels and probabilities
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
    try:
        model = model.to(device)
        model.eval()
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        
        # Ensure inputs are on the correct device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Apply sigmoid to get probabilities for multi-label classification
            probs = torch.sigmoid(outputs.logits)[0].cpu().numpy()
            
    except RuntimeError as e:
        # Fallback to CPU if GPU/MPS fails (common with MPS)
        if device != "cpu":
            print(f"Prediction on {device} failed, falling back to CPU. Error: {e}")
            return predict_event(text, tokenizer, model, id2label, threshold, device="cpu")
        raise e

    # Get all labels above threshold
    predicted_labels = []
    predicted_probs = []
    
    for idx, prob in enumerate(probs):
        # Determine threshold for this class
        if isinstance(threshold, (dict, list, np.ndarray)):
            if isinstance(threshold, dict):
                cls_threshold = threshold.get(id2label[idx], 0.5)
            else:
                cls_threshold = threshold[idx]
        else:
            cls_threshold = threshold
            
        if prob >= cls_threshold:
            predicted_labels.append(id2label[idx])
            predicted_probs.append(float(prob))
    
    # Sort by probability (descending)
    if predicted_labels:
        sorted_pairs = sorted(zip(predicted_labels, predicted_probs), 
                            key=lambda x: x[1], reverse=True)
        predicted_labels, predicted_probs = zip(*sorted_pairs)
        predicted_labels = list(predicted_labels)
        predicted_probs = list(predicted_probs)
    
    # Get all probabilities for reference
    all_probs = {id2label[idx]: float(prob) for idx, prob in enumerate(probs)}
    
    return {
        "text": text,
        "predicted_labels": predicted_labels,
        "predicted_probs": predicted_probs,
        "all_probabilities": all_probs,
        "num_labels": len(predicted_labels)
    }


if __name__ == "__main__":
    # 1) point this to your checkpoint or final model dir
    MODEL_DIR = "../outputs/sentifm/checkpoint-924"  # adjust if needed

    try:
        tokenizer, model, id2label = load_model_and_tokenizer(MODEL_DIR)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please update MODEL_DIR to point to a valid model checkpoint")
        exit(1)

    # === 1) Sanity check on real test data ===
    try:
        from src.data.load_sentifm import load_sentivent_event_type
        
        base_dir = Path(__file__).resolve().parents[1]  # src/
        data_dir = base_dir / "data"
        
        dataset, label2id_data, id2label_data = load_sentivent_event_type(str(data_dir))
        test_ds = dataset["test"]

        print("=== Sanity check on real test examples ===")
        print("Showing first 5 examples with at least one label\n")
        
        count = 0
        for ex in test_ds:
            # Check if example has any labels
            if sum(ex["labels"]) > 0:
                pred = predict_event(ex["text"], tokenizer, model, id2label)
                
                # Get true labels
                true_labels = [id2label_data[idx] for idx, val in enumerate(ex["labels"]) if val == 1]
                
                print("-" * 80)
                print(f"Text: {ex['text'][:100]}...")
                print(f"True labels: {true_labels}")
                print(f"Predicted labels: {pred['predicted_labels']}")
                print(f"Predicted probs: {[f'{p:.3f}' for p in pred['predicted_probs']]}")
                
                count += 1
                if count >= 5:
                    break
    except Exception as e:
        print(f"Could not load test data: {e}")
        print("Skipping sanity check on test data\n")

    # === 2) Your custom examples ===
    print("\n" + "="*80)
    print("=== Custom examples ===")
    print("="*80 + "\n")
    
    examples = [
        "Apple beats Wall Street expectations on strong iPhone demand.",
        "Tesla to acquire a battery startup in a 400 million dollar deal.",
        "SEC opens investigation into Coinbase over unregistered securities.",
        "Unilever was criticised by shareholders at its annual meeting.",
        "Microsoft announces new dividend and share buyback program.",
        "Amazon stock drops 5% after missing revenue targets.",
    ]

    for text in examples:
        pred = predict_event(text, tokenizer, model, id2label)
        print("-" * 80)
        print(f"Text: {pred['text']}")
        print(f"Predicted labels: {pred['predicted_labels']}")
        print(f"Probabilities: {[f'{p:.3f}' for p in pred['predicted_probs']]}")
        print(f"Number of labels: {pred['num_labels']}")
        print()
