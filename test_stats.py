import numpy as np
import pytest
from stats import calculate_stats

def test_calculate_stats_perfect_prediction():
    """Test with perfect predictions (all correct)"""
    normal_pred = np.array([0, 0, 0])  # All normal predicted correctly
    abnormal_pred = np.array([1, 1, 1])  # All abnormal predicted correctly
    
    stats = calculate_stats(normal_pred, abnormal_pred)
    
    assert stats['confusion_matrix'].tolist() == [[3, 0], [0, 3]]
    assert stats['precision'] == 1.0
    assert stats['recall'] == 1.0
    assert stats['f1_score'] == 1.0

def test_calculate_stats_all_wrong():
    """Test with all wrong predictions"""
    normal_pred = np.array([1, 1, 1])  # All normal predicted as abnormal
    abnormal_pred = np.array([0, 0, 0])  # All abnormal predicted as normal
    
    stats = calculate_stats(normal_pred, abnormal_pred)
    
    assert stats['confusion_matrix'].tolist() == [[0, 3], [3, 0]]
    assert stats['precision'] == 0.0
    assert stats['recall'] == 0.0
    with pytest.raises(ZeroDivisionError):
        _ = stats['f1_score']

def test_calculate_stats_mixed_predictions():
    """Test with mixed predictions"""
    normal_pred = np.array([0, 0, 1, 1])  # 2 correct, 2 wrong
    abnormal_pred = np.array([1, 0, 1, 0])  # 2 correct, 2 wrong
    
    stats = calculate_stats(normal_pred, abnormal_pred)
    
    assert stats['confusion_matrix'].tolist() == [[2, 2], [2, 2]]
    assert stats['precision'] == 0.5
    assert stats['recall'] == 0.5
    assert stats['f1_score'] == 0.5

def test_calculate_stats_empty_arrays():
    """Test with empty arrays"""
    normal_pred = np.array([])
    abnormal_pred = np.array([])
    
    with pytest.raises(ZeroDivisionError):
        _ = calculate_stats(normal_pred, abnormal_pred) 