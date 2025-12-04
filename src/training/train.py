import torch
from torch import nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss
        
        if self.alpha is not None:
            # Apply alpha weighting
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            focal_loss = self.alpha * focal_loss
            
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class MultilabelTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        # Initialize Focal Loss
        self.focal_loss = FocalLoss(alpha=class_weights, gamma=2.0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss

def train(model, tokenizer, dataset):
    # Calculate class weights for imbalance handling
    print("Calculating class weights for Focal Loss...")
    train_labels = np.array(dataset["train"]["labels"])
    num_positives = np.sum(train_labels, axis=0)
    num_negatives = len(train_labels) - num_positives
    
    # Avoid division by zero
    num_positives = np.clip(num_positives, 1, None)
    
    # For Focal Loss, alpha is typically set to inverse frequency or similar
    # We'll use the dampened weights we found effective
    pos_weights = torch.tensor(np.sqrt(num_negatives / num_positives), dtype=torch.float)
    
    print("Class weights (alpha for Focal Loss):")
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
        save_strategy="no",  # Don't save checkpoints during training
        load_best_model_at_end=False,  # We'll just use the final model
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
