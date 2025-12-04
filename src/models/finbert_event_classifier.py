from transformers import AutoModelForSequenceClassification


def load_model(num_labels, label2id, id2label):
    print(f"Loading ProsusAI/finbert with {num_labels} labels...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "ProsusAI/finbert",
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # <-- this is the key
    )

    # Small sanity print
    classifier = getattr(model, "classifier", None)
    if classifier is not None:
        try:
            print("Classifier weight shape:", classifier.weight.shape)
        except AttributeError:
            pass

    return model
