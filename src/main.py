from data.load_sentifm import load_sentivent_event_type
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from training.train import train
from evaluation.evaluate import evaluate

def main():
    # Load dataset with multi-label binary vectors
    dataset, label2id, id2label = load_sentivent_event_type(
        "src/data/"  # directory containing dataset_event_type.tsv and type_classes_multilabelbinarizer.json
    )

    print(f"Loaded dataset with {len(label2id)} labels")
    print(f"Train examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")

    # Use DeBERTa-v3-Large as requested
    model_name = "microsoft/deberta-v3-large"
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    # Tokenize the dataset
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )

    # Create model for multi-label classification
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    print("\nModel initialized for multi-label classification")
    print(f"Number of labels: {len(label2id)}")
    print(f"Labels: {list(label2id.keys())}")

    # Train the model
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)
    trainer = train(model, tokenizer, dataset)

    # Evaluate the model
    print("\n" + "="*80)
    print("Starting evaluation...")
    print("="*80)
    macro_f1, model_dir, best_thresholds = evaluate(trainer, dataset, id2label)

    print("\n" + "="*80)
    print("Training and evaluation complete!")
    print(f"Macro F1 Score: {macro_f1:.3f}")
    print(f"Best Thresholds: Per-class optimized")
    print(f"Model saved to: {model_dir}")
    print("="*80)

    # Sample predictions on custom examples
    print("\n" + "="*80)
    print("Sample Predictions on Custom Examples")
    print("="*80)
    
    from predict.predict import predict_event
    
    custom_examples = [
        "Apple beats Wall Street expectations on strong iPhone demand.",
        "Tesla to acquire a battery startup in a 400 million dollar deal.",
        "SEC opens investigation into Coinbase over unregistered securities.",
        "Microsoft announces new dividend and share buyback program.",
        "Amazon stock drops 5% after missing revenue targets.",
        "Google CEO steps down amid restructuring plans.",
    ]
    
    for text in custom_examples:
        pred = predict_event(text, tokenizer, trainer.model, id2label, threshold=best_thresholds)
        print("-" * 80)
        print(f"Text: {text}")
        print(f"Predicted labels: {pred['predicted_labels']}")
        print(f"Probabilities: {[f'{p:.3f}' for p in pred['predicted_probs']]}")

    return trainer, macro_f1, model_dir

if __name__ == "__main__":
    main()