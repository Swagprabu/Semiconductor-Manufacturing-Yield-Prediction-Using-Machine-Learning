# Semiconductor Manufacturing Yield Prediction Using Machine Learning

## Overview

This is a **Capstone Project** that focuses on predicting semiconductor manufacturing yield (Pass/Fail classification) using advanced machine learning techniques. The project leverages sensor data from manufacturing processes to build predictive models that can help optimize production quality and reduce defects.

## Project Objective

Manufacturing yield prediction is critical in the semiconductor industry. This project aims to:
- Analyze sensor data from semiconductor manufacturing processes
- Identify patterns and relationships in the data
- Build accurate machine learning models to predict whether a batch will pass or fail quality checks
- Compare multiple algorithms to determine the best performing model
- Provide actionable insights for improving manufacturing efficiency

## Dataset

**File:** `sensor-data.csv`

The dataset contains sensor readings and manufacturing parameters from semiconductor production lines with a target variable `Pass/Fail` indicating the quality status of each batch.

### Data Characteristics:
- Multiple sensor measurements capturing various aspects of the manufacturing process
- Temporal and operational features
- Class labels indicating pass/fail outcomes
- Real-world data with missing values and noise

## Project Structure

```
├── Cap-2.ipynb                    # Main Jupyter notebook with full analysis and modeling
├── sensor-data.csv                # Raw sensor data from manufacturing process
├── best_model.pkl                 # Best performing trained model (serialized)
├── naive_bayes.pkl                # Naive Bayes model
├── random_forest.pkl              # Random Forest model
├── svm.pkl                         # Support Vector Machine model
└── README.md                       # Project documentation
```

## Methodology

### 1. **Data Exploration & Audit**
- Analyzed dataset shape and structure
- Identified missing values and their patterns
- Examined target variable distribution
- Detected duplicate records and constant columns
- Statistical profiling of features

### 2. **Data Cleaning & Preprocessing**
- Removed timestamp columns (temporal feature engineering)
- Dropped columns with >50% missing values
- Handled missing values using median imputation
- Removed all-NaN columns
- Feature normalization and standardization

### 3. **Feature Engineering**
- Extracted features from sensor data
- Created meaningful derived features
- Feature scaling for algorithm compatibility
- Feature selection based on importance scores

### 4. **Model Development**
Multiple machine learning algorithms were trained and evaluated:

- **Naive Bayes**: Probabilistic classifier based on Bayes' theorem
- **Random Forest**: Ensemble method combining multiple decision trees
- **Support Vector Machine (SVM)**: Powerful classifier for non-linear relationships

### 5. **Model Evaluation**
- Train-test split validation
- Cross-validation for robust performance assessment
- Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion matrix analysis
- Model comparison and selection

### 6. **Best Model Selection**
The best performing model was serialized and saved as `best_model.pkl` for production use.

## Key Technologies & Libraries

```
Python 3.x
pandas       - Data manipulation and analysis
numpy        - Numerical computing
scikit-learn - Machine learning algorithms and evaluation metrics
matplotlib   - Data visualization
seaborn      - Statistical data visualization
pickle       - Model serialization
```

## Installation & Setup

### Prerequisites:
- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

### Install Required Libraries:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage

### Running the Analysis:

1. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook Cap-2.ipynb
   ```

2. **Execute Cells Sequentially:**
   - Run data loading and exploration cells first
   - Execute data cleaning pipeline
   - Run feature engineering steps
   - Train models sequentially
   - Evaluate and compare results

### Making Predictions with the Best Model:

```python
import pickle
import pandas as pd

# Load the best model
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Prepare your data (ensure same preprocessing as training)
# X_new = your_sensor_data

# Make predictions
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

## Results & Performance

The project evaluates three different algorithms:
- **Naive Bayes**: Fast, interpretable baseline model
- **Random Forest**: Good balance of accuracy and interpretability
- **SVM**: High accuracy on non-linear problems

The **best model** (selected based on cross-validation performance) is saved and ready for deployment.

## Key Findings

- Sensor data contains valuable patterns predictive of manufacturing yield
- Feature engineering from raw sensor readings improves model performance
- Ensemble methods (Random Forest) typically outperform individual classifiers
- Proper data preprocessing significantly impacts model accuracy
- Multiple algorithms should be compared to find the optimal solution

## Model Artifacts

### Trained Models:
- `best_model.pkl` - The optimized model for production predictions
- `naive_bayes.pkl` - Naive Bayes classifier
- `random_forest.pkl` - Random Forest classifier
- `svm.pkl` - Support Vector Machine classifier

These can be loaded using Python's `pickle` module for making predictions on new data.

## Future Improvements

- **Advanced Feature Engineering**: Domain-specific feature creation with manufacturing expertise
- **Deep Learning**: Explore neural networks for complex pattern recognition
- **Time Series Analysis**: Incorporate temporal dependencies in sensor readings
- **Real-Time Prediction**: Deploy models for live manufacturing process monitoring
- **Hyperparameter Optimization**: Use GridSearch or Bayesian optimization for tuning
- **Explainability**: Implement SHAP or LIME for model interpretability
- **Production Pipeline**: Create automated ETL and model serving infrastructure

## Performance Metrics

The models are evaluated on:
- **Accuracy**: Overall correctness of predictions
- **Precision**: Correctness of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Trade-off between true positive and false positive rates

## Project Insights

1. **Data Quality**: Manufacturing data requires careful preprocessing due to sensor noise
2. **Class Distribution**: The Pass/Fail distribution affects model selection
3. **Algorithm Comparison**: No single algorithm dominates; context matters
4. **Production Ready**: Saved models enable easy deployment for real-time predictions

## Author

**Project:** Semiconductor Manufacturing Yield Prediction - Capstone Project 2

## License

This project is for educational purposes. Feel free to use and modify for your own learning and research.

## Contact & Support

For questions or issues related to this project, please reach out or open an issue in the repository.

---

**Last Updated:** 2026  
**Project Status:** Complete ✓
