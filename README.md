# KNN

This project implements a simple Multi-class K‑Nearest Neighbors (KNN) algorithm from scratch.

We classify tree species based on their geographic coordinates by finding their closest neighbors and using majority voting.

## Algorithm Overview
- Compute Euclidean distance between the query point and all other points
- Select the K nearest neighbors and sort them in ascending order
- Perform majority voting
    - If there is a tie, the class of the closest neighbor is chosen

## Model Evaluation:
We evaluate the classifier using:
1. **Accuracy**  

$$
\text{Accuracy} = \frac{TP}{All Samples}
$$

2. **Precision**  

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

3. **Recall**  

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

## Libraries:
- `pandas`: for reading csv and creatring a dataframe
- `kagglehub`: to download the latest dataset

## Dataset
[Urban Tree Health Monitoring Dataset](https://www.kaggle.com/datasets/khushikyad001/urban-tree-health-monitoring-dataset)