from datasets import DatasetDict

def downsample_no_event(dataset, max_noevent=3000, noevent_label="NoEvent"):
    """
    Downsample NoEvent examples in the train split.
    Assumes dataset["train"]["label"] is still string labels.
    """
    train = dataset["train"]
    labels = train["label"]

    no_idx = [i for i, lab in enumerate(labels) if lab == noevent_label]
    other_idx = [i for i, lab in enumerate(labels) if lab != noevent_label]

    print(f"Original train size: {len(train)}")
    print(f"NoEvent count       : {len(no_idx)}")
    print(f"Event count         : {len(other_idx)}")

    keep_no = no_idx[:max_noevent]
    keep_indices = keep_no + other_idx

    new_train = train.select(keep_indices)

    print(f"New train size      : {len(new_train)}")
    print(f"New NoEvent count   : {sum(1 for i in keep_indices if labels[i] == noevent_label)}")

    return DatasetDict({"train": new_train, "test": dataset["test"]})
