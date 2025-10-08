# CDTree: Causal Discovery with Tree-Based Models

## Overview

CDTree is a Python framework for performing causal discovery in tabular data. It utilizes an ensemble of specialized decision trees (an Unsupervised Causal Forest) to uncover potential causal relationships between variables in an unsupervised manner. This approach is particularly useful for generating causal hypotheses from observational data without prior knowledge of the causal graph.

This approach of mine is new, and requires significant testing. Results can often deviate with hyperparameters, particularly beta.

There are two ways around this for now:
1. If you have a dataset, and you want to find casual relationships, create a new dataset based on the expectations of that dataset, using its properties. Tune the hyperparameters until the relationships are realized, and see if it holds true for the original data.
2. This require further experimentation, but you can try adjusting beta by a tiny amount until changes in leaf node counts result in lesser changes regarding which side is dominant. Following this, you may wish to adjust the leaf node count until you get diminishing returns.

Currently, this approach does not perform well for non-linear relationships at this moment, pending possible improvements. However, you can create polynomial representations, and the model will find these relationships.

## Core Components

- **`UnsupervisedCausalTree`**: The fundamental building block of the framework. It is a single decision tree designed to identify causal influence by measuring the change in impurity of other features when a split is made on a particular feature.

- **`UnsupervisedCausalForest`**: An ensemble of `UnsupervisedCausalTree`s. By aggregating the findings of many trees, the forest provides a more robust and stable estimation of the causal impact matrix.

- **`ConfounderProcessor`**: A post-processing tool that refines the output of the causal discovery models. It helps to resolve confounding relationships by analyzing reciprocal influences in the impact matrix.

## How It Works

The core idea is that a feature has a causal influence on other features if splitting the data based on the causal feature leads to a reduction in the impurity (e.g., variance or Gini impurity) of the affected features in the resulting child nodes. The algorithm quantifies this change as the causal impact.

The forest model builds a multitude of these trees on different subsets of the data and averages their findings to produce a final, aggregated causal impact matrix that is less prone to noise and overfitting than a single tree.

## Splitting Criterion

Note: All x values are represented as Y; the z-score normalized continuous values as their own y values, and categories where (1 - HHI) is their impurity, also known as Gini Impurity.

The core of the causal discovery tree lies in its splitting criterion, which aims to identify splits that maximize the causal impact. For each potential split on a feature, the algorithm calculates a score based on the gains in impurity for all other features.

Let $G_{pos}$ be the sum of positive gains and $G_{neg}$ be the sum of absolute negative gains across all *other* features (i.e., excluding the splitting feature $F_{split}$) for a given split:

$$
G_{pos} = \sum_{i \neq F_{split} \text{ s.t. } Gain_i > 0} Gain_i
$$

$$
G_{neg} = \sum_{i \neq F_{split} \text{ s.t. } Gain_i < 0} |Gain_i|
$$

The splitting score for a candidate split is then calculated as:

$$
Score = (1 - G_{pos} - G_{neg}) \times (G_{pos} + G_{neg})^{\beta}
$$

Where $\beta$ is a hyperparameter that controls the emphasis on the magnitude of gains.

The $Gain_i$ for a feature $i$ is defined as the reduction in impurity achieved by the split:

$$
Gain_i = Impurity_{parent}(F_i) - \left( p_{left} \times Impurity_{left}(F_i) + p_{right} \times Impurity_{right}(F_i) \right)
$$

Here, $p_{left}$ and $p_{right}$ are the proportions of samples in the left and right child nodes, respectively.

The impurity measure used depends on the feature type:
- For **continuous** features, variance is used: $Impurity(F_i) = Var(F_i)$.
- For **categorical** features, Gini impurity is used: $Impurity(F_i) = 1 - \sum_{k=1}^{C} p_k^2$, where $p_k$ is the proportion of samples belonging to class $k$.

## Getting Started

### Installation

To use this framework, you can install the dependencies directly from the `pyproject.toml` file or by cloning the repository.

```bash
# To install dependencies from pyproject.toml
pip install .

# Alternatively, clone the repository
# git clone https://github.com/your-repo/CDTree.git
# cd CDTree
# pip install .
```

I might release on PyPI pending further testing of this project.

### Usage

The primary entry point for this framework is the `UnsupervisedCausalForest` class. You can fit it on your data (a NumPy array) and then call the `get_impact_matrix()` method to get the results.

## Examples

The `src/` directory contains several examples of how to use the framework:

- **`example_usage_forest.py`**: Demonstrates the standard workflow using `UnsupervisedCausalForest`.
- **`example_usage_tree.py`**: Shows how to use a single `UnsupervisedCausalTree`.
- **`example_usage_tree_v2.py`**: Illustrates how to manually apply the `ConfounderProcessor` to the output of a single tree.

---

*This project was created with the assistance of Gemini.*
