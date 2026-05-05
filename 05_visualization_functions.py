# Cell 5: Visualization Functions

def plot_training_history(histories, model_names):
    """
    Plot training and validation curves for all models.
    
    Args:
        histories: list of history dictionaries
        model_names: list of model names
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    for hist, name in zip(histories, model_names):
        axes[0].plot(hist['train_loss'], label=f'{name} - Train', linestyle='--')
        axes[0].plot(hist['val_loss'], label=f'{name} - Val')
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracy
    for hist, name in zip(histories, model_names):
        axes[1].plot(hist['train_acc'], label=f'{name} - Train', linestyle='--')
        axes[1].plot(hist['val_acc'], label=f'{name} - Val')
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix'):
    """
    Plot confusion matrix.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        class_names: list of class names
        title: plot title
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return cm


def compare_model_performance(results_dict):
    """
    Create a comparison table and bar plot of model performance.
    
    Args:
        results_dict: dictionary with model names as keys and 
                     (accuracy, f1_score) as values
    """
    # Create comparison table
    print("\n" + "="*70)
    print(f"{'Model':<25} {'Accuracy (%)':<20} {'F1-Score':<20}")
    print("="*70)
    
    model_names = []
    accuracies = []
    f1_scores = []
    
    for model_name, (acc, f1) in results_dict.items():
        print(f"{model_name:<25} {acc:<20.2f} {f1:<20.4f}")
        model_names.append(model_name)
        accuracies.append(acc)
        f1_scores.append(f1)
    
    print("="*70 + "\n")
    
    # Create bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    bars1 = axes[0].bar(model_names, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 100])
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom', fontweight='bold')
    
    # F1-Score comparison
    bars2 = axes[1].bar(model_names, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    axes[1].set_ylabel('F1-Score', fontsize=12)
    axes[1].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def analyze_per_class_performance(y_true, y_pred, class_names, model_name):
    """
    Analyze and visualize per-class performance.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        class_names: list of class names
        model_name: name of the model
    """
    # Get classification report
    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)
    
    # Extract per-class metrics
    classes = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for class_name in class_names:
        if class_name in report:
            classes.append(class_name)
            precisions.append(report[class_name]['precision'])
            recalls.append(report[class_name]['recall'])
            f1_scores.append(report[class_name]['f1-score'])
    
    # Create visualization
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precisions, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recalls, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Per-Class Performance - {model_name}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.show()
    
    return report
