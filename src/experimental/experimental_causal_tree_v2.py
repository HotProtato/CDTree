
import numpy as np
from dataclasses import dataclass, field
from typing import Any
from scipy.spatial.distance import pdist, squareform

# --- Kernel Functions ---
def rbf_kernel(X, gamma=1.0):
    """
    Computes the Radial Basis Function (RBF) kernel.
    X should be a 2D array of shape (n_samples, 1).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    sq_dists = squareform(pdist(X, 'sqeuclidean'))
    return np.exp(-gamma * sq_dists)

def discrete_kernel(X):
    """
    Computes a discrete kernel. It's 1 if values are the same, 0 otherwise.
    X should be a 2D array of shape (n_samples, 1).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return (X == X.T).astype(float)

# --- Core Tree Model ---
@dataclass(slots=True)
class _Node:
    feature_index: int = None
    threshold: Any = None
    is_leaf: bool = False
    value: np.ndarray = None
    left: '_Node' = None
    right: '_Node' = None
    impact_matrix: np.ndarray = None
    n_samples: int = 0

class ExperimentalCausalTreeV2:
    """
    An experimental causal discovery tree with a dynamic impurity metric.
    """
    def __init__(self, max_leaf_nodes=None, min_samples_leaf=5, alpha=1.0, beta=1.0, cat_threshold=10, min_gain_threshold=0.0):
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.beta = beta
        self.cat_threshold = cat_threshold
        self.min_gain_threshold = min_gain_threshold
        self.root = None
        self.feature_types = []
        self.feature_names = []
        self.max_impurity = 0.0
        self.num_splits = 0

    def fit(self, X, feature_names=None):
        if feature_names and len(feature_names) == X.shape[1]:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'F{i}' for i in range(X.shape[1])]
        self.n_samples, self.n_features = X.shape
        self.feature_types = self._determine_feature_types(X)
        self.num_splits = 0
        
        # Calculate max_impurity at the root
        root_impurities = [self._calculate_impurity(X[:, i], self.feature_types[i]) for i in range(self.n_features)]
        self.max_impurity = sum(root_impurities)

        self.root = self._build_tree(X)

    def _build_tree(self, X, depth=0):
        n_samples_node = len(X)
        is_max_leaves_reached = self.max_leaf_nodes is not None and self.num_splits + 1 >= self.max_leaf_nodes

        if is_max_leaves_reached or n_samples_node < self.min_samples_leaf:
            return _Node(is_leaf=True, value=self._leaf_value(X), n_samples=n_samples_node)

        best_split = self._find_best_split(X)

        if best_split is None:
            return _Node(is_leaf=True, value=self._leaf_value(X), n_samples=n_samples_node)

        self.num_splits += 1
        left_indices, right_indices = best_split['left_indices'], best_split['right_indices']
        left_child = self._build_tree(X[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], depth + 1)

        return _Node(
            feature_index=best_split['feature_index'],
            threshold=best_split['threshold'],
            left=left_child,
            right=right_child,
            impact_matrix=best_split['impact_matrix'],
            n_samples=n_samples_node
        )

    def _find_best_split(self, X):
        n_samples, n_features = X.shape
        best_score = -float('inf')
        best_split = None

        parent_impurities = {i: self._calculate_impurity(X[:, i], self.feature_types[i]) for i in range(n_features)}

        for feature_index in range(n_features):
            unique_values = np.unique(X[:, feature_index])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2 if self.feature_types[feature_index] == 'continuous' else unique_values

            for threshold in thresholds:
                if self.feature_types[feature_index] == 'continuous':
                    left_indices = X[:, feature_index] <= threshold
                    right_indices = X[:, feature_index] > threshold
                else:
                    left_indices = X[:, feature_index] == threshold
                    right_indices = X[:, feature_index] != threshold
                
                X_left, X_right = X[left_indices], X[right_indices]

                if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
                    continue

                gains = np.zeros(n_features)
                for i in range(n_features):
                    if i == feature_index: continue
                    gains[i] = self._calculate_gain(parent_impurities[i], X_left[:, i], X_right[:, i], self.feature_types[i])
                
                cross_gain_agree = np.sum(gains)
                
                if self.max_impurity == 0:
                    score = 0
                else:
                    normalized_cross_gain = np.abs(cross_gain_agree) / self.max_impurity
                    score = self.alpha * (1 - normalized_cross_gain) * (np.abs(cross_gain_agree) ** self.beta)

                if score > best_score:
                    best_score = score
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'score': score,
                        'gains': gains,
                        'left_indices': left_indices,
                        'right_indices': right_indices
                    }
        
        if best_split is not None:
            if best_split['score'] < self.min_gain_threshold:
                return None
            impact_matrix = np.zeros((n_features, n_features))
            impact_matrix[best_split['feature_index'], :] = best_split['gains']
            best_split['impact_matrix'] = impact_matrix

        return best_split

    def _calculate_impurity(self, y, feature_type):
        n = len(y)
        if n < 2:
            return 0.0

        if feature_type == 'categorical':
            kernel_matrix = discrete_kernel(y)
        else: # continuous
            # Ensure the array is numeric before calculating RBF kernel
            y_numeric = y.astype(float)
            kernel_matrix = rbf_kernel(y_numeric)
            
        sum_K = np.sum(kernel_matrix)
        return 1 - (1 / (n**2)) * sum_K

    def _calculate_gain(self, parent_impurity, left_child, right_child, feature_type):
        n = len(left_child) + len(right_child)
        if n == 0:
            return 0.0
        p = len(left_child) / n
        if p == 0 or p == 1:
            return 0.0
        
        impurity_left = self._calculate_impurity(left_child, feature_type)
        impurity_right = self._calculate_impurity(right_child, feature_type)
        
        return parent_impurity - (p * impurity_left + (1 - p) * impurity_right)

    def _determine_feature_types(self, X):
        feature_types = []
        for i in range(X.shape[1]):
            col = X[:, i]
            # Check if column is numeric before using isnan
            if np.issubdtype(col.dtype, np.number):
                unique_values = np.unique(col[~np.isnan(col)])
            else:
                # For non-numeric types (like object/string), we can't use isnan.
                unique_values = np.unique(col.astype(str))
            
            if len(unique_values) <= self.cat_threshold:
                feature_types.append('categorical')
            else:
                feature_types.append('continuous')
        return feature_types

    def get_impact_matrix(self, aggregate_symmetric: bool = False):
        """
        Aggregates the impact matrices from all nodes in the tree.

        Note on Directionality: The model directly discovers causal influence. However,
        for very strong causal links (A -> B), the algorithm may sometimes identify the
        reverse direction (B -> A) as being stronger. It is often useful to consider
        the relationship as a strong, but potentially undirected, link.

        Args:
            aggregate_symmetric (bool): If True, returns a symmetric matrix where the
                impact between A and B is the sum of (A -> B) and (B -> A), representing
                the total association strength absent of a single direction.
                Defaults to False.

        Returns:
            np.ndarray: The aggregated causal impact matrix.
        """
        aggregated_matrix = np.zeros((self.n_features, self.n_features))
        def _traverse(node):
            nonlocal aggregated_matrix
            if not node.is_leaf:
                aggregated_matrix += node.impact_matrix * (node.n_samples / self.n_samples)
                _traverse(node.left)
                _traverse(node.right)
        if self.root: _traverse(self.root)

        if aggregate_symmetric:
            return aggregated_matrix + aggregated_matrix.T
        else:
            return aggregated_matrix

    def get_n_leaves(self):
        def _count_leaves(node):
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return _count_leaves(node.left) + _count_leaves(node.right)
        return _count_leaves(self.root)

    def _leaf_value(self, X):
        """Calculates the value for a leaf node.
        - Mean for continuous features.
        - Mode for categorical features.
        """
        leaf_values = []
        for i in range(X.shape[1]):
            col = X[:, i]
            feature_type = self.feature_types[i]
            
            if feature_type == 'continuous':
                # Ensure column is numeric before taking mean
                leaf_values.append(np.mean(col.astype(float)))
            else: # categorical
                # Find the mode (most frequent value)
                unique, counts = np.unique(col, return_counts=True)
                mode = unique[np.argmax(counts)]
                leaf_values.append(mode)
                
        return np.array(leaf_values, dtype=object)

