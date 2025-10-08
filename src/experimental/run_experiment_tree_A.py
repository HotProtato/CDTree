
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.experimental.experimental_causal_tree import UnsupervisedCausalTree
from src.model.cd_tree import ConfounderProcessor # ConfounderProcessor is still in cd_tree

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

def run_experiment_A():
    """
    Runs an experiment on Dataset A using the experimental UnsupervisedCausalTree.
    """
    print("Running experiment on Dataset A with experimental Causal Tree...")
    
    # --- 1. Generate Data ---
    data = generate_scenario_A(n_samples=200, random_state=42)
    X = data.values
    feature_names = data.columns.tolist()
    print(f"Features: {feature_names}")

    # --- 2. Instantiate and Fit the Experimental Tree ---
    tree_config = {
        'beta': 1.0,
        'max_leaf_nodes': 11,
        'max_score_threshold': 0.5,
        'min_samples_leaf': 10
    }

    print("\nInstantiating and fitting the Experimental Unsupervised Causal Tree...")
    tree = UnsupervisedCausalTree(**tree_config)
    tree.fit(X, feature_names=feature_names)

    # --- 3. Retrieve and Visualize the Causal Impact Matrix ---
    print("\nRetrieving the causal impact matrix from the experimental tree...")
    impact_matrix = tree.get_impact_matrix()
    impact_df = pd.DataFrame(impact_matrix, index=feature_names, columns=feature_names)

    print("\n--- Causal Discovery Results (Experimental Tree on Dataset A) ---")
    print("Causal Impact Matrix:")
    print(impact_df.round(3))

    # --- Optional: Process with ConfounderProcessor and Visualize ---
    print("\nProcessing the matrix with ConfounderProcessor (optional)...")
    processor = ConfounderProcessor()
    refined_matrix = processor.process_matrix(impact_matrix)
    refined_df = pd.DataFrame(refined_matrix, index=feature_names, columns=feature_names)
    
    print("\nRefined Causal Impact Matrix (Experimental Tree on Dataset A):")
    print(refined_df.round(3))

    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(refined_df, annot=True, fmt=".3f", cmap='viridis')
        plt.title('Refined Causal Impact Matrix (Experimental Tree on Dataset A)')
        plt.ylabel('Causal Feature (Impactor)')
        plt.xlabel('Affected Feature (Impactee)')
        
        heatmap_path = 'experimental_tree_A_heatmap.png'
        plt.savefig(heatmap_path)
        print(f"\nSuccess! Heatmap saved to: {heatmap_path}")

    except ImportError:
        print("\nWarning: Could not generate heatmap. Please install matplotlib and seaborn (`pip install matplotlib seaborn`) to get a visualization.")

if __name__ == "__main__":
    run_experiment_A()
