import numpy as np

def calculate_stats(normal_predictions, abnormal_predictions):
    """Calculate confusion matrix and derived statistics.
    
    Args:
        normal_predictions: Array of predictions for normal samples (0=normal, 1=anomaly)
        abnormal_predictions: Array of predictions for abnormal samples (0=normal, 1=anomaly)
    
    Returns:
        dict: Dictionary containing confusion matrix and statistics
    """
    confusion_matrix = np.zeros((2, 2))
    confusion_matrix[0,0] = np.sum(normal_predictions == 0)  # TN
    confusion_matrix[0,1] = np.sum(normal_predictions == 1)  # FN
    confusion_matrix[1,0] = np.sum(abnormal_predictions == 0)  # FP
    confusion_matrix[1,1] = np.sum(abnormal_predictions == 1)  # TP

    precision = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[1,0])
    recall = confusion_matrix[1,1] / (confusion_matrix[1,1] + confusion_matrix[0,1])
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return {
        'confusion_matrix': confusion_matrix,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }

def print_stats(stats, title="Results:", params=None):
    """Print formatted statistics.
    
    Args:
        stats: Dictionary containing confusion matrix and statistics
        title: Optional title for the output
        params: Optional dictionary of model parameters to display
    """
    print(f"\n{title}")
    
    if params:
        for param_name, param_value in params.items():
            print(f"{param_name}: {param_value}")
    
    confusion_matrix = stats['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                  Actual Normal    Actual Anomaly")
    print(f"Predicted Normal     {confusion_matrix[0,0]:.0f}             {confusion_matrix[0,1]:.0f}")
    print(f"Predicted Anomaly    {confusion_matrix[1,0]:.0f}             {confusion_matrix[1,1]:.0f}")

    print(f"\nPrecision: {stats['precision']:.3f}")
    print(f"Recall: {stats['recall']:.3f}")
    print(f"F1 score: {stats['f1_score']:.3f}")