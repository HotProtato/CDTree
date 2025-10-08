
import numpy as np
from dataclasses import dataclass, field

# --- Data Processing Component ---
class ConfounderProcessor:
    """
    Processes an impact matrix to refine relationships based on reciprocal influence.
    """
    def process_matrix(self, impact_matrix: np.ndarray) -> np.ndarray:
        if impact_matrix.ndim != 2 or impact_matrix.shape[0] != impact_matrix.shape[1]:
            raise ValueError("Input matrix must be a square 2D numpy array.")

        num_features = impact_matrix.shape[0]
        refined_matrix = np.copy(impact_matrix)

        for i in range(num_features):
            for j in range(i + 1, num_features):
                if refined_matrix[i, j] != 0 and refined_matrix[j, i] != 0:
                    if abs(refined_matrix[i, j]) < abs(refined_matrix[j, i]):
                        refined_matrix[j, i] += refined_matrix[i, j] ** 2
                        refined_matrix[i, j] = 0
                    else:
                        refined_matrix[i, j] += refined_matrix[j, i] ** 2
                        refined_matrix[j, i] = 0
        return refined_matrix

# --- Core Tree Model ---
@dataclass(slots=True)
class _Node:
    feature_index: int = None
    threshold: float = None
    is_leaf: bool = False
    value: np.ndarray = None
    left: '_Node' = None
    right: '_Node' = None
    impact_matrix: np.ndarray = None
    n_samples: int = 0

class UnsupervisedCausalTree:
    """
    A single decision tree for unsupervised causal discovery.
    This is the internal building block for the UnsupervisedCausalForest.
    """
    def __init__(self, max_depth=None, min_samples_leaf=5, max_leaf_nodes=None, beta=1.0, min_score_threshold=0.0, max_score_threshold=1.0, max_features=1.0):
        self.max_depth = max_depth if max_depth is not None else float('inf')
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.beta = beta
        self.min_score_threshold = min_score_threshold
        self.max_score_threshold = max_score_threshold
        self.max_features = max_features
        self.root = None
        self.feature_types = []
        self.feature_names = []

    def fit(self, X, feature_names=None):
        if feature_names and len(feature_names) == X.shape[1]:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'F{i}' for i in range(X.shape[1])]
        self.n_samples, self.n_features = X.shape
        self.num_splits = 0
        self.feature_types = self._determine_feature_types(X)
        self.root = self._build_tree(X)

    def _build_tree(self, X, depth=0):
        n_samples_node = len(X)
        is_max_leaves_reached = self.max_leaf_nodes is not None and self.num_splits + 1 >= self.max_leaf_nodes

        if depth >= self.max_depth or n_samples_node < self.min_samples_leaf or is_max_leaves_reached:
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
        best_score = -np.inf
        best_split = None
        n_samples, n_features = X.shape
        parent_impurities = {i: self._calculate_impurity(X[:, i], self.feature_types[i]) for i in range(n_features)}

        n_features_subspace = int(n_features * self.max_features)
        subspace_indices = np.random.choice(n_features, n_features_subspace, replace=False)

        for feature_index in subspace_indices:
            unique_values = np.unique(X[:, feature_index])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2 if self.feature_types[feature_index] == 'continuous' else unique_values

            for threshold in thresholds:
                if self.feature_types[feature_index] == 'continuous':
                    left_indices, right_indices = X[:, feature_index] <= threshold, X[:, feature_index] > threshold
                else:
                    left_indices, right_indices = X[:, feature_index] == threshold, X[:, feature_index] != threshold
                
                X_left, X_right = X[left_indices], X[right_indices]

                if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
                    continue

                gains = np.zeros(n_features)
                for i in range(n_features):
                    if i == feature_index: continue
                    gains[i] = self._calculate_gain(parent_impurities[i], X_left[:, i], X_right[:, i], self.feature_types[i])

                sum_pos_gain = np.sum(gains[gains > 0])
                sum_abs_neg_gain = np.sum(np.abs(gains[gains < 0]))
                score = (1 - sum_pos_gain - sum_abs_neg_gain) * (sum_pos_gain + sum_abs_neg_gain)**self.beta

                if score > best_score:
                    best_score = score
                    impact_matrix = np.zeros((n_features, n_features))
                    impact_matrix[feature_index, :] = gains
                    best_split = {
                        'feature_index': feature_index, 'threshold': threshold, 'score': score,
                        'impact_matrix': impact_matrix, 'left_indices': left_indices, 'right_indices': right_indices
                    }

        if best_split and (best_split['score'] < self.min_score_threshold or best_split['score'] > self.max_score_threshold):
            return None

        return best_split

    def _calculate_impurity(self, y, feature_type):
        if len(y) < 2: return 0.0
        if feature_type == 'categorical':
            _, counts = np.unique(y, return_counts=True)
            return 1 - np.sum((counts / len(y))**2)
        else:
            return np.var(y)

    def _calculate_gain(self, parent_impurity, left_child, right_child, feature_type):
        p = len(left_child) / (len(left_child) + len(right_child))
        if p == 0 or p == 1: return 0.0
        return parent_impurity - (p * self._calculate_impurity(left_child, feature_type) + (1 - p) * self._calculate_impurity(right_child, feature_type))

    def _determine_feature_types(self, X, cat_threshold=10):
        return ['categorical' if len(np.unique(X[:, i][~np.isnan(X[:, i])])) <= cat_threshold else 'continuous' for i in range(X.shape[1])]
        
    def _leaf_value(self, X):
        return np.mean(X, axis=0)

    def get_impact_matrix(self):
        aggregated_matrix = np.zeros((self.n_features, self.n_features))
        def _traverse(node):
            nonlocal aggregated_matrix
            if not node.is_leaf:
                aggregated_matrix += node.impact_matrix * (node.n_samples / self.n_samples)
                _traverse(node.left)
                _traverse(node.right)
        if self.root: _traverse(self.root)
        return aggregated_matrix
