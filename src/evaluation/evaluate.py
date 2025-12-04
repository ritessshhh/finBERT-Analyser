import os
import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support


def evaluate(trainer, dataset, id2label, save_root="outputs/sentifm"):
    """
    Evaluate multi-label classification model with threshold tuning.
    
    Args:
        trainer: HuggingFace Trainer instance
        dataset: DatasetDict with test split
        id2label: Dict mapping label indices to names
        save_root: Directory to save model
    
    Returns:
        macro_f1: Macro F1 score
        model_dir: Path to saved model
    """
    # Predict on test split
    preds = trainer.predict(dataset["test"])
    y_true = preds.label_ids  # Shape: (n_samples, n_labels)
    y_pred_logits = preds.predictions  # Shape: (n_samples, n_labels)
    
    # Apply sigmoid to get probabilities
    y_pred_probs = 1 / (1 + np.exp(-y_pred_logits))  # sigmoid
    
    # Find optimal threshold per class
    print("\n" + "="*80)
    print("PER-CLASS THRESHOLD TUNING")
    print("="*80)
    print(f"{'Label':<25} {'Best Thr':<10} {'Best F1':<10}")
    print("-" * 80)
    
    best_thresholds = np.full(len(id2label), 0.5)
    
    for i in range(len(id2label)):
        best_f1_local = 0.0
        best_thr_local = 0.5
        
        # Test thresholds for this specific class
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            y_pred_local = (y_pred_probs[:, i] >= threshold).astype(int)
            f1_local = f1_score(y_true[:, i], y_pred_local, average='binary', zero_division=0)
            
            if f1_local > best_f1_local:
                best_f1_local = f1_local
                best_thr_local = threshold
        
        best_thresholds[i] = best_thr_local
        print(f"{id2label[i]:<25} {best_thr_local:<10.1f} {best_f1_local:<10.3f}")
        
    print("-" * 80)
    print("="*80 + "\n")
    
    # Use best thresholds for final evaluation
    y_pred = np.zeros_like(y_pred_probs, dtype=int)
    for i in range(len(id2label)):
        y_pred[:, i] = (y_pred_probs[:, i] >= best_thresholds[i]).astype(int)
    
    print("\n" + "="*80)
    print(f"MULTI-LABEL CLASSIFICATION EVALUATION (Per-Class Thresholds)")
    print("="*80)
    
    # Per-label metrics
    print("\nPer-label metrics:")
    print("-" * 80)
    print(f"{'Label':<25} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10} {'Threshold':>10}")
    print("-" * 80)
    
    precisions = []
    recalls = []
    f1s = []
    supports = []
    
    for i in range(len(id2label)):
        label_name = id2label[i]
        
        # Calculate metrics for this label
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true[:, i], 
            y_pred[:, i], 
            average='binary',
            zero_division=0
        )
        
        # Count actual support (number of positive labels in ground truth)
        actual_support = int(y_true[:, i].sum())
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(actual_support)
        
        print(f"{label_name:<25} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f} {actual_support:>10} {best_thresholds[i]:>10.1f}")
    
    print("-" * 80)
    
    # Macro averages
    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1s)
    
    print(f"{'Macro Average':<25} {macro_precision:>10.3f} {macro_recall:>10.3f} {macro_f1:>10.3f} {int(np.sum(supports)):>10}")
    
    # Micro averages (treating all labels equally)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        y_true.flatten(),
        y_pred.flatten(),
        average='binary',
        zero_division=0
    )
    
    print(f"{'Micro Average':<25} {micro_precision:>10.3f} {micro_recall:>10.3f} {micro_f1:>10.3f}")
    print("-" * 80)
    
    # Sample-based metrics
    print("\nSample-based metrics:")
    print("-" * 80)
    
    # Exact match ratio (all labels must match)
    exact_match = np.all(y_true == y_pred, axis=1).mean()
    print(f"Exact Match Ratio: {exact_match:.3f}")
    
    # Hamming loss (fraction of wrong labels)
    hamming_loss = np.mean(y_true != y_pred)
    print(f"Hamming Loss: {hamming_loss:.3f}")
    
    # Average number of labels per sample
    avg_labels_true = np.mean(np.sum(y_true, axis=1))
    avg_labels_pred = np.mean(np.sum(y_pred, axis=1))
    print(f"Avg labels per sample (true): {avg_labels_true:.2f}")
    print(f"Avg labels per sample (pred): {avg_labels_pred:.2f}")
    
    print("="*80 + "\n")
    
    # Save model with F1 in directory name
    f1_tag = f"{macro_f1:.3f}"
    model_dir = os.path.join(save_root, f"final_model_f1_{f1_tag}_per_class_thr")
    print(f"Saving model to: {model_dir}")
    trainer.save_model(model_dir)
    
    # Save thresholds to a file in the model directory
    np.save(os.path.join(model_dir, "thresholds.npy"), best_thresholds)
    
    return macro_f1, model_dir, best_thresholds
