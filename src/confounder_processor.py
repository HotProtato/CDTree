import numpy as np

class ConfounderProcessor:
    """
    Processes an impact matrix to refine relationships based on reciprocal influence.
    If there's a reciprocal relationship (A->B and B->A), the weaker relationship's
    value is squared and added to the stronger one, and the weaker is set to zero.
    """
    def process_matrix(self, impact_matrix: np.ndarray) -> np.ndarray:
        """
        Applies the post-processing logic to the impact matrix.

        :param impact_matrix: The raw impact matrix (numpy array).
        :return: The refined impact matrix.
        """
        if impact_matrix.ndim != 2 or impact_matrix.shape[0] != impact_matrix.shape[1]:
            raise ValueError("Input matrix must be a square 2D numpy array.")

        num_features = impact_matrix.shape[0]
        refined_matrix = np.copy(impact_matrix)

        for i in range(num_features):
            for j in range(num_features):
                if i == j:  # Skip self-impact
                    continue

                # Check for reciprocal relationship
                if refined_matrix[i, j] != 0 and refined_matrix[j, i] != 0:
                    if abs(refined_matrix[i, j]) < abs(refined_matrix[j, i]):
                        # (i, j) is weaker than (j, i)
                        refined_matrix[j, i] += refined_matrix[i, j] ** 2
                        refined_matrix[i, j] = 0
                    else:
                        # (j, i) is weaker or equal to (i, j)
                        refined_matrix[i, j] += refined_matrix[j, i] ** 2
                        refined_matrix[j, i] = 0
        
        return refined_matrix

    # The logic is whichever direction is strongest is most likely the direction of impact. The weaker direction
    # is awarded to the stronger direction after getting the root.