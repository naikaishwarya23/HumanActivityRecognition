# Human Activity Recognition Using Smartphones

## Overview
This project implements a machine learning pipeline for human activity recognition using smartphone sensor data. It encompasses data preprocessing, model building, and evaluation, leveraging deep learning with TensorFlow/Keras and classical machine learning techniques with Scikit-learn.

## Key Components

### 1. Library Installation and Imports
- **Installed Libraries**:
  - `TensorFlow`: For deep learning model development.
  - `Keras-Tuner`: For hyperparameter tuning of neural networks.
- **Imported Libraries**:
  - **Data Processing**: `numpy`, `pandas`
  - **Visualization**: `matplotlib`, `seaborn`
  - **Machine Learning**:
    - `tensorflow.keras`: Modules for building neural networks (e.g., `Sequential`, `Dense`, `LSTM`).
    - `sklearn`: Modules for preprocessing (`MinMaxScaler`, `LabelEncoder`), classification (`RandomForestClassifier`, `VotingClassifier`, `SVC`), and metrics evaluation.

### 2. Data Loading
- **Source**: Training and test datasets are loaded from external URLs, with the dataset uploaded to the cloud for easy access.
- **Dataset Link**: [Kaggle - Human Activity Recognition with Smartphones](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones/data)
- **Dataset Details**:
  - **Training Set**: 7,352 rows, 563 columns.
  - **Test Set**: 2,947 rows, 563 columns.
  - **Features**: Mostly numerical (`float64`), with one categorical column (`Activity`).

### 3. Potential Preprocessing Steps
- **Normalization**: Applied using `MinMaxScaler` to scale numerical features.
- **Label Encoding**: Categorical `Activity` labels encoded using `LabelEncoder`.

### 4. Model Building
- **Neural Networks**: Built using `tensorflow.keras.models.Sequential` and layers from `tensorflow.keras.layers`.
- **Ensemble Methods**: Includes `RandomForestClassifier`, `VotingClassifier`, and `SVC` for robust classification.
- **Optimization**: Utilizes callbacks like `EarlyStopping` to prevent overfitting during training.

### 5. Metrics and Evaluation
- **Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix.
- **Reports**: Supports detailed classification reports for performance analysis.

### 6. Conclusion
- The **ensemble voting classifier with tuned hyperparameters** emerged as the best model due to:
  - Higher accuracy.
  - Balanced performance across all activity classes.
  - Fewer misclassifications.
- This makes it ideal for real-world applications requiring consistent, high-quality predictions.

## Insights
- **Pipeline Objective**: Develop a robust model for multi-class activity classification using neural networks and ensemble methods.
- **Scalability**: Integration of `Keras-Tuner` enables hyperparameter optimization for improved performance.
- **Preprocessing**: Comprehensive tools ensure effective feature scaling and label encoding.
- **Potential Improvements**: Incorporate sequential models like Recurrent Neural Networks (RNNs) or Long Short-Term Memory (LSTM) networks to capture temporal dependencies and transitions in the sensor data.

## Getting Started

### Prerequisites
- Python 3.x
- Required libraries: `tensorflow`, `keras-tuner`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/naikaishwarya23/HumanActivityRecognition.git
   cd HumanActivityRecognition
