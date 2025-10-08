
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from src.model.cd_forest import UnsupervisedCausalForest

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

def run_example():
    """
    Demonstrates the full workflow for the UnsupervisedCausalForest model:
    1. Generate synthetic data.
    2. Instantiate and fit the forest with proven hyperparameters.
    3. Retrieve and visualize the final causal impact matrix.
    """
    # --- 1. Generate Data ---
    print("Generating synthetic data for Scenario A...")
    data = generate_scenario_A(n_samples=200)
    
    X = data.values
    feature_names = data.columns.tolist()
    print(f"Features: {feature_names}")

    # --- 2. Instantiate and Fit the Forest ---
    # Using a 'golden' configuration discovered through HPO.
    golden_config = {
        'beta': 1.1,
        'max_leaf_nodes': 8,
        'max_score_threshold': 0.8,
        'min_samples_leaf': 5
    }

    print("\nInstantiating and fitting the Unsupervised Causal Forest...")
    forest = UnsupervisedCausalForest(n_estimators=100, **golden_config)
    forest.fit(X, feature_names=feature_names)

    # --- 3. Retrieve and Visualize the Causal Impact Matrix ---
    print("\nRetrieving and refining the causal impact matrix...")
    refined_matrix = forest.get_impact_matrix(refine=True)

    print("\n--- Causal Discovery Results ---")
    print("Refined Causal Impact Matrix:")
    print(refined_matrix.round(3))

    # Generate and save a heatmap for easy interpretation
    try:
        plt.figure(figsize=(12, 10))
        sns.heatmap(refined_matrix, annot=True, fmt=".3f", cmap='viridis')
        plt.title('Final Causal Impact Matrix (Scenario A)')
        plt.ylabel('Causal Feature (Impactor)')
        plt.xlabel('Affected Feature (Impactee)')
        
        heatmap_path = '../../heatmaps/causal_discovery_heatmap.png'
        plt.savefig(heatmap_path)
        print(f"\nSuccess! Heatmap saved to: {heatmap_path}")

    except ImportError:
        print("\nWarning: Could not generate heatmap. Please install matplotlib and seaborn (`pip install matplotlib seaborn`) to get a visualization.")

if __name__ == "__main__":
    run_example()
