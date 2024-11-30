import pytest
import numpy as np
import pandas as pd
from ngram import NGramTree, NGramModel

@pytest.fixture
def simple_tree():
    tree = NGramTree(max_depth=2)
    tree.add_sequence(['A', 'B', 'C', 'B', 'A'])
    return tree

@pytest.fixture
def simple_model():
    model = NGramModel(max_ngram_size=2)
    model.fit(['A', 'B', 'C', 'B', 'A'])
    return model

def test_ngram_tree_initialization():
    tree = NGramTree(max_depth=3)
    assert tree.max_depth == 3
    assert tree.tree == {}

def test_ngram_tree_add_sequence():
    tree = NGramTree(max_depth=2)
    tree.add_sequence(['A', 'B', 'C'])
    
    # Check if the tree structure is correct
    assert 'A' in tree.tree
    assert 'B' in tree.tree['A']
    assert tree.tree['A']['B'] == 1
    
    assert 'B' in tree.tree
    assert 'C' in tree.tree['B']
    assert tree.tree['B']['C'] == 1

def test_ngram_tree_get_counts(simple_tree):
    # Test getting counts for existing context
    counts = simple_tree.get_counts(pd.Series(['A']))
    assert 'B' in counts
    assert counts['B'] == 1

    # Test getting counts for non-existing context
    counts = simple_tree.get_counts(pd.Series(['X']))
    assert counts == {}

def test_ngram_model_initialization():
    model = NGramModel(max_ngram_size=3)
    assert model.max_ngram_size == 3
    assert isinstance(model.tree, NGramTree)

def test_ngram_model_fit():
    model = NGramModel(max_ngram_size=2)
    sequence = ['A', 'B', 'C', 'B', 'A']
    model.fit(sequence)
    
    # Verify the tree contains expected patterns
    counts = model.tree.get_counts(pd.Series(['A']))
    assert 'B' in counts

def test_ngram_model_fit_blocks():
    model = NGramModel(max_ngram_size=2)
    blocks = [
        ['A', 'B', 'C'],
        ['B', 'C', 'A']
    ]
    model.fit_blocks(blocks)
    
    # Verify patterns from both blocks are present
    counts = model.tree.get_counts(pd.Series(['B']))
    assert 'C' in counts

def test_ngram_model_predict(simple_model):
    sequence = ['A', 'B', 'C']
    anomalies = simple_model.predict(sequence, top_k=1)
    assert isinstance(anomalies, np.ndarray)
    assert len(anomalies) == len(sequence)

def test_ngram_model_predict_block(simple_model):
    block = ['A', 'B', 'C']
    result = simple_model.predict_block(block, top_k=1, anomaly_count_threshold=0.5)
    assert not result

def test_empty_sequence():
    tree = NGramTree(max_depth=2)
    tree.add_sequence([])
    assert tree.tree == {}

def test_single_token_sequence():
    tree = NGramTree(max_depth=2)
    tree.add_sequence(['A'])
    assert tree.tree == {} 

def test_ngram_model_unigram():
    # Test model with max_ngram_size=1 (unigram model)
    model = NGramModel(max_ngram_size=1)
    sequence = ['A', 'B', 'A', 'A', 'C']
    model.fit(sequence)
    
    # Verify counts are correct for individual tokens
    counts = model.tree.get_counts(pd.Series([]))
    assert counts['A'] == 3  # 'A' appears 3 times
    assert counts['B'] == 1  # 'B' appears once
    assert counts['C'] == 1  # 'C' appears once
    
    # Verify predictions work with unigram model
    anomalies = model.predict(['A', 'B'], top_k=1)
    assert isinstance(anomalies, np.ndarray)
    assert len(anomalies) == 2 