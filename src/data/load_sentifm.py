import json
import pandas as pd
from pathlib import Path
from collections import Counter
from datasets import Dataset, DatasetDict
import ast


def load_sentivent_event_type(data_dir=None):
    """
    Load the event type dataset from TSV file with multi-label binary vectors.
    
    Args:
        data_dir: Directory containing dataset_event_type.tsv and type_classes_multilabelbinarizer.json
    
    Returns:
        dataset: HuggingFace DatasetDict with 'train' and 'test' splits
        label2id: Dict mapping label names to indices
        id2label: Dict mapping indices to label names
    """
    if data_dir is None:
        base_dir = Path(__file__).resolve().parents[2]
        data_dir = base_dir / "src" / "data"
    else:
        data_dir = Path(data_dir)
    
    # Load label classes
    label_file = data_dir / "type_classes_multilabelbinarizer.json"
    with open(label_file, "r") as f:
        label_classes = json.load(f)
    
    # Create label mappings
    label2id = {label: idx for idx, label in enumerate(label_classes)}
    id2label = {idx: label for idx, label in enumerate(label_classes)}
    
    # Load TSV data
    tsv_file = data_dir / "dataset_event_type.tsv"
    df = pd.read_csv(tsv_file, sep="\t")
    
    # Parse the labels column (string representation of list to actual list)
    df["labels"] = df["labels"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Split into train and test based on 'dataset' column
    train_df = df[df["dataset"] == "silver"].copy()
    test_df = df[df["dataset"] == "gold"].copy()
    
    # Prepare examples (convert labels to float for BCEWithLogitsLoss)
    train_examples = [
        {"text": row["text"], "labels": [float(x) for x in row["labels"]]}
        for _, row in train_df.iterrows()
    ]
    
    test_examples = [
        {"text": row["text"], "labels": [float(x) for x in row["labels"]]}
        for _, row in test_df.iterrows()
    ]
    
    # Create HuggingFace datasets
    train_ds = Dataset.from_list(train_examples)
    test_ds = Dataset.from_list(test_examples)
    
    dataset = DatasetDict({"train": train_ds, "test": test_ds})
    
    return dataset, label2id, id2label


def load_sentifm(path=None):
    """
    Backward compatibility wrapper.
    Returns a HuggingFace DatasetDict with fields: text, labels (list of ints).
    """
    if path is None:
        base_dir = Path(__file__).resolve().parents[2]
        data_dir = base_dir / "src" / "data"
    else:
        data_dir = Path(path).parent if Path(path).is_file() else Path(path)
    
    dataset, label2id, id2label = load_sentivent_event_type(data_dir)
    return dataset


def show_label_stats(dataset, id2label):
    """Show statistics about label distribution in the dataset."""
    
    def count_labels(examples):
        """Count how many times each label appears."""
        label_counts = Counter()
        for labels in examples:
            for idx, val in enumerate(labels):
                if val == 1:
                    label_counts[id2label[idx]] += 1
        return label_counts
    
    train_labels = dataset["train"]["labels"]
    test_labels = dataset["test"]["labels"]
    
    train_counts = count_labels(train_labels)
    test_counts = count_labels(test_labels)
    
    print("\nLabel distribution (TRAIN):")
    for label, count in train_counts.most_common():
        print(f"{label:25s} {count:5d}")
    
    print("\nLabel distribution (TEST):")
    for label, count in test_counts.most_common():
        print(f"{label:25s} {count:5d}")
    
    # Show multi-label statistics
    train_label_counts = [sum(labels) for labels in train_labels]
    test_label_counts = [sum(labels) for labels in test_labels]
    
    print("\nMulti-label statistics (TRAIN):")
    print(f"  Avg labels per example: {sum(train_label_counts) / len(train_label_counts):.2f}")
    print(f"  Max labels per example: {max(train_label_counts)}")
    print(f"  Examples with 0 labels: {train_label_counts.count(0)}")
    
    print("\nMulti-label statistics (TEST):")
    print(f"  Avg labels per example: {sum(test_label_counts) / len(test_label_counts):.2f}")
    print(f"  Max labels per example: {max(test_label_counts)}")
    print(f"  Examples with 0 labels: {test_label_counts.count(0)}")


if __name__ == "__main__":
    # Test the data loading
    dataset, label2id, id2label = load_sentivent_event_type()
    
    print("Dataset loaded successfully!")
    print(f"\nNumber of labels: {len(label2id)}")
    print(f"Label classes: {list(label2id.keys())}")
    
    print(f"\nTrain examples: {len(dataset['train'])}")
    print(f"Test examples: {len(dataset['test'])}")
    
    # Show some examples
    print("\n" + "="*80)
    print("Sample examples from TRAIN:")
    print("="*80)
    for i in range(3):
        example = dataset["train"][i]
        active_labels = [id2label[idx] for idx, val in enumerate(example["labels"]) if val == 1]
        print(f"\nExample {i+1}:")
        print(f"Text: {example['text'][:100]}...")
        print(f"Labels: {active_labels}")
        print(f"Label vector: {example['labels']}")
    
    # Show statistics
    show_label_stats(dataset, id2label)
