import numpy as np
import pandas as pd
from typing import Dict, Any

class NGramTree:
    """
    A tree-based data structure for storing and counting n-gram sequences.
    
    The tree stores sequences of tokens where each path from root to leaf
    represents an n-gram. The leaf nodes contain counts of how many times
    that specific n-gram sequence appears in the training data.
    
    Parameters:
        max_depth (int): Maximum length of n-grams to store in the tree
    
    Attributes:
        tree (Dict[Any, Any]): Nested dictionary representing the n-gram tree structure
            where inner nodes are dictionaries and leaf nodes are counts
    """
    def __init__(self, max_depth: int):
        self.max_depth = max_depth
        # Use nested defaultdict to automatically create new dicts for new paths
        self.tree: Dict[Any, Any] = dict()
        
    def add_sequence(self, sequence: list):
        """
        Add a sequence to the n-gram tree and update counts.
        
        For each possible n-gram in the sequence, traverses the tree to the appropriate
        position and increments the count for that n-gram pattern.
        
        Parameters:
            sequence (list): List of tokens to process into n-grams
        """
        depth = -1 # if self.max_depth is 0
        for i in range(len(sequence) - self.max_depth+1):
            current = self.tree
            # Build path through tree except for last token
            for depth in range(self.max_depth - 1):  # Stop one before last
                token = sequence[i + depth]
                if token not in current:
                    current[token] = dict()
                current = current[token]
            
            # Handle the last token specially
            final_token = sequence[i + depth + 1]
            if final_token not in current:
                current[final_token] = 0
            current[final_token] += 1
    
  
    def get_counts(self, context: pd.Series) -> Dict[Any, float]:
        """
        Retrieve the frequency counts of tokens that follow a given context sequence.
        
        Parameters:
            context (pd.Series): A sequence of tokens representing the context/history
                               to look up in the n-gram tree
        
        Returns:
            Dict[Any, float]: A dictionary mapping each possible next token to its
                             frequency count. Returns empty dict if context not found
                             in the tree.
        
        Note:
            The method traverses the tree using the context sequence and returns
            the leaf node containing counts. If any token in the context is not
            found, returns an empty dictionary.
        """
        current = self.tree

        # Navigate to the context
        for token in context:
            if token not in current:
                return {}
            current = current[token]

        return current

class NGramModel:
    """
    An N-gram based anomaly detection model for sequence data.
    
    This model builds a tree-based n-gram model to learn sequence patterns
    and detect anomalies based on frequency distributions.
    
    Parameters:
        max_ngram_size (int): Maximum length of n-grams to consider
    """
    def __init__(self, max_ngram_size):
        self.max_ngram_size = max_ngram_size
        self.tree = NGramTree(max_ngram_size)

    def fit_blocks(self, blocks):
        """
        Train the model on multiple sequences of events.
        
        Parameters:
            blocks (list): List of event sequences, where each sequence is a list
                          of events to learn patterns from
        
        Note:
            This method processes multiple sequences by adding each sequence to
            the n-gram tree independently, allowing the model to learn patterns
            from multiple separate event streams.
        """
        for block in blocks:
            self.tree.add_sequence(block)

    def fit(self, events):
        """
        Train the model on a sequence of events.
        
        Parameters:
            events (list): List of event sequences to learn patterns from
        """
        self.tree.add_sequence(events)

    def predict_block(self, block, top_k, anomaly_count_threshold):
        """
        Predict if a block of events contains anomalous patterns.
        
        Parameters:
            block (list): Sequence of events to analyze
            top_k (int): Number of most frequent next events to consider normal
            anomaly_count_threshold (float): Threshold ratio of anomalies to mark block as anomalous
            
        Returns:
            bool: True if block is considered anomalous, False otherwise
        """
        anomalies = self.predict(block, top_k)
        anomaly_count = np.sum(anomalies)

        prediction = anomaly_count / len(block) > anomaly_count_threshold
        return prediction
    
    def predict(self, sequence, top_k):
        """
        Identify anomalous events within a sequence.
        
        Parameters:
            sequence (list): Sequence of events to analyze
            top_k (int): Number of most frequent next events to consider normal
            anomaly_count_threshold (float): Threshold for anomaly detection
            
        Returns:
            numpy.ndarray: Binary array marking anomalous events (1) and normal events (0)
        """
        anomalies = np.zeros(len(sequence))
        for i in range(len(sequence)-self.max_ngram_size):
            token_index = i+self.max_ngram_size-1
            context = sequence[i:token_index]

            dist = self.tree.get_counts(context)

            max_counts = dict(sorted(dist.items(), key=lambda x: x[1], reverse=True)[:top_k])
            if sequence[token_index] not in max_counts:
                anomalies[token_index] = 1
        
        return anomalies