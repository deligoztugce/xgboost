# XGBoost Classifier on Iris Dataset

This project demonstrates the use of the XGBoost classifier to classify the famous Iris dataset. The Iris dataset is a multi-class classification problem where the goal is to predict the species of the Iris flower based on the flower's attributes.

## Dataset
The Iris dataset contains 150 samples of iris flowers, each with 4 features:
- Sepal length
- Sepal width
- Petal length
- Petal width

There are 3 target classes:
- Setosa
- Versicolor
- Virginica

## Code Overview

### 1. Loading the Data:
The dataset is loaded using `sklearn.datasets.load_iris()`.

### 2. Preprocessing:
- The data is split into training and test sets using `train_test_split()`.
- Features are standardized using `StandardScaler()`.

### 3. Model Training:
An `XGBClassifier` from the `xgboost` library is trained using the training data.

### 4. Model Evaluation:
The trained model is evaluated on the test data, and the following metrics are displayed:
- **Accuracy**
- **Precision, Recall, F1-score** (using `classification_report()`)
- **Confusion Matrix**

### 5. Feature Importance:
The importance of each feature in the model's prediction is visualized using a bar plot.

## Results

- **Accuracy**: The model achieves 100% accuracy on the test set.
- **Classification Report**: Shows perfect precision, recall, and F1-score for each class.
- **Confusion Matrix**: The model predicts all test instances correctly, resulting in a diagonal matrix.

## Visualizations

- **Feature Importance**: A bar plot shows the importance of each feature (sepal length, sepal width, petal length, petal width) in predicting the species.
- **Confusion Matrix**: A heatmap displays the confusion matrix with true vs. predicted labels.

## Conclusion
This project demonstrates a simple yet effective classification task using the XGBoost algorithm. The model performs perfectly on the Iris dataset with 100% accuracy.


## Project Overview

We will:
1. Load the Iris dataset.
2. Preprocess the data by splitting it into training and testing sets and standardizing the features.
3. Train an XGBoost classifier.
4. Evaluate the model's performance using accuracy and a detailed classification report.
5. Visualize the feature importances and the confusion matrix.

## Requirements

To run this project, you'll need to install the following libraries:

```bash
pip install xgboost scikit-learn pandas matplotlib seaborn
