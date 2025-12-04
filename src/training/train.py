import torch
from torch import nn
from transformers import Trainer, TrainingArguments
import numpy as np

class MultilabelTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            # Ensure weights are on the same device as logits
            weights = self.class_weights.to(logits.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=weights)
        else:
            loss_fct = nn.BCEWithLogitsLoss()
            
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train(model, tokenizer, dataset):
    # Calculate class weights for imbalance handling
    print("Calculating class weights for imbalanced loss...")
    train_labels = np.array(dataset["train"]["labels"])
    num_positives = np.sum(train_labels, axis=0)
    num_negatives = len(train_labels) - num_positives
    
    # Avoid division by zero
    num_positives = np.clip(num_positives, 1, None)
    
    # Calculate pos_weight = sqrt(number_of_negatives / number_of_positives)
    # Using sqrt dampens the effect of extreme imbalance, improving precision
    pos_weights = torch.tensor(np.sqrt(num_negatives / num_positives), dtype=torch.float)
    
    print("Class weights (pos_weight) [Dampened with SQRT]:")
    print(pos_weights)

    print("Creating TrainingArguments...")

    args = TrainingArguments(
        output_dir="outputs/sentifm",
        num_train_epochs=10,
        per_device_train_batch_size=8,  # Reduced for RoBERTa Large
        per_device_eval_batch_size=16,  # Reduced for RoBERTa Large
        gradient_accumulation_steps=2,  # Accumulate gradients to simulate batch size 16
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.02,
        logging_dir="outputs/sentifm/logs",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        lr_scheduler_type="cosine",
    )

    print("Initializing MultilabelTrainer...")
    trainer = MultilabelTrainer(
        class_weights=pos_weights,
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )

    print("Calling trainer.train()...")
    trainer.train()
    print("Training finished.")

    return trainer
