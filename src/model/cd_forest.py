import numpy as np
import pandas as pd
from src.model.cd_tree import UnsupervisedCausalTree, ConfounderProcessor

class UnsupervisedCausalForest:
    """
    Discovers causal relationships in tabular data using an ensemble of specialized decision trees.
    """

    def __init__(self, n_estimators=100, max_features='sqrt', **tree_kwargs):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.tree_kwargs = tree_kwargs
        self.trees = []
        self.feature_names = []

    def fit(self, X, feature_names=None):
        self.trees = []
        if feature_names and len(feature_names) == X.shape[1]:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'F{i}' for i in range(X.shape[1])]

        n_samples, n_features = X.shape

        if self.max_features == 'sqrt':
            max_features_val = int(np.sqrt(n_features))
        elif isinstance(self.max_features, float):
            max_features_val = int(n_features * self.max_features)
        else:
            max_features_val = n_features

        self.tree_kwargs['max_features'] = max_features_val / n_features

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X[indices]
            tree = UnsupervisedCausalTree(**self.tree_kwargs)
            tree.fit(X_sample, feature_names=self.feature_names)
            self.trees.append(tree)

    def get_impact_matrix(self, refine=False):
        if not self.trees:
            raise RuntimeError("The forest has not been fitted yet.")

        all_matrices = [tree.get_impact_matrix() for tree in self.trees]
        aggregated_matrix = np.mean(all_matrices, axis=0)

        if refine:
            processor = ConfounderProcessor()
            aggregated_matrix = processor.process_matrix(aggregated_matrix)

        return pd.DataFrame(aggregated_matrix, index=self.feature_names, columns=self.feature_names)
