# Machine Learning Model Evaluator

A Python-based evaluation pipeline that tests and compares multiple machine learning classification models. The script processes standard datasets through different training/testing data splits to extract accuracy percentages, error rates, and confusion matrices.

### Technologies & Libraries Used
* **Python 3**
* **scikit-learn** (Dataset loading, Model training, Metrics extraction)

### Models Evaluated
The script compares the performance of the following classifiers:
* **Decision Tree** (J48/CART via `DecisionTreeClassifier`)
* **Naive Bayes** (`GaussianNB`)
* **Backpropagation / Neural Network** (via `MLPClassifier` with 2000 max iterations)

### Datasets Tested
* **Iris Dataset:** Standard botanical classification.
* **Wine Dataset:** Chemical analysis classification.

### Data Partitions Evaluated
To test model robustness, the pipeline evaluates each model under two different data split conditions:
* 50% Training / 50% Testing
* 80% Training / 20% Testing

## How to Run

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
