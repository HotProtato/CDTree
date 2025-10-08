
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.model.cd_tree import UnsupervisedCausalTree, ConfounderProcessor

def generate_scenario_A(n_samples=100, random_state=42):
    """
    Scenario A: Simple Linear Confounder
    X2 (confounder) -> X1 (treatment) and X2 -> Y (outcome)
    X1 (treatment) -> Y (outcome)
    """
    if random_state: np.random.seed(random_state)
    
    X2 = np.random.normal(0, 1, n_samples) # Confounder
    X3 = np.random.normal(0, 1, n_samples) # Noise feature
    
    X1 = 0.5 * X2 + np.random.normal(0, 0.5, n_samples) # X1 influenced by X2
    Y = 2 * X1 + 1.5 * X2 + np.random.normal(0, 1, n_samples) # Y influenced by X1 and X2
    
    df = pd.DataFrame({'X1_treatment': X1, 'X2_confounder': X2, 'X3_noise': X3, 'Y_outcome': Y})
    return df

def run_example_tree_v2():
    """
    Demonstrates using the ConfounderProcessor with a single UnsupervisedCausalTree:
    1. Generate synthetic data.
    2. Instantiate and fit the tree.
    3. Retrieve the raw impact matrix.
    4. Process the matrix with ConfounderProcessor.
    5. Visualize both matrices.
    """
    # --- 1. Generate Data ---
    print("Generating synthetic data for Scenario A...")
    data = generate_scenario_A(n_samples=200)
    
    X = data.values
    feature_names = data.columns.tolist()
    print(f"Features: {feature_names}")

    # --- 2. Instantiate and Fit the Tree ---
    tree_config = {
        'beta': 1.1,
        'max_leaf_nodes': 8,
        'max_score_threshold': 0.8,
        'min_samples_leaf': 5
    }

    print("\nInstantiating and fitting the Unsupervised Causal Tree...")
    tree = UnsupervisedCausalTree(**tree_config)
    tree.fit(X, feature_names=feature_names)

    # --- 3. Retrieve the Causal Impact Matrix ---
    print("\nRetrieving the causal impact matrix from the tree...")
    impact_matrix = tree.get_impact_matrix()
    impact_df = pd.DataFrame(impact_matrix, index=feature_names, columns=feature_names)

    print("\n--- Causal Discovery Results (Single Tree) ---")
    print("Raw Causal Impact Matrix:")
    print(impact_df.round(3))

    # --- 4. Process with ConfounderProcessor ---
    print("\nProcessing the matrix with ConfounderProcessor...")
    processor = ConfounderProcessor()
    refined_matrix = processor.process_matrix(impact_matrix)
    refined_df = pd.DataFrame(refined_matrix, index=feature_names, columns=feature_names)
    
    print("\nRefined Causal Impact Matrix:")
    print(refined_df.round(3))


    # --- 5. Visualize Both Matrices ---
    try:
        fig, axes = plt.subplots(1, 2, figsize=(24, 10))
        
        # Raw Matrix
        sns.heatmap(impact_df, annot=True, fmt=".3f", cmap='viridis', ax=axes[0])
        axes[0].set_title('Raw Causal Impact Matrix (Single Tree)')
        axes[0].set_ylabel('Causal Feature (Impactor)')
        axes[0].set_xlabel('Affected Feature (Impactee)')
        
        # Refined Matrix
        sns.heatmap(refined_df, annot=True, fmt=".3f", cmap='viridis', ax=axes[1])
        axes[1].set_title('Refined Causal Impact Matrix (with ConfounderProcessor)')
        axes[1].set_ylabel('Causal Feature (Impactor)')
        axes[1].set_xlabel('Affected Feature (Impactee)')
        
        heatmap_path = '../../heatmaps/causal_discovery_tree_v2_heatmaps.png'
        plt.savefig(heatmap_path)
        print(f"\nSuccess! Heatmaps saved to: {heatmap_path}")

    except ImportError:
        print("\nWarning: Could not generate heatmaps. Please install matplotlib and seaborn (`pip install matplotlib seaborn`) to get a visualization.")

if __name__ == "__main__":
    run_example_tree_v2()
