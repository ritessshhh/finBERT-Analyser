def build_label_maps(dataset):
    labels = sorted(set(dataset["train"]["label"]))
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def encode_labels(example, label2id):
    example["label"] = label2id[example["label"]]
    return example
