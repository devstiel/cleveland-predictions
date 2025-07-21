# Cleveland Heart Disease Machine Learning Project

## 📋 Overview

This project implements multiple machine learning algorithms to predict heart disease using the Cleveland Heart Disease dataset. The goal is to build and compare various classification models to accurately identify patients at risk of heart disease based on medical attributes.

## 🎯 Objective


## 📊 Dataset

**Source**: Cleveland Heart Disease Dataset (`processed.cleveland.data`)

**Description**: This dataset contains 304 instances with 14 attributes related to heart disease diagnosis.

### Features:
- **age**: Age in years
- **sex**: Gender (1 = male, 0 = female)
- **cp**: Chest pain type (1-4)
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise
- **slope**: Slope of peak exercise ST segment (1-3)
- **ca**: Number of major vessels colored by fluoroscopy (0-3)
- **thal**: Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)

### Target Variable:
- **target**: Heart disease presence (0 = no disease, 1 = disease)

## 🛠️ Machine Learning Algorithms Implemented

### 1. **K-Nearest Neighbors (KNN)**
- **Purpose**: Classification based on similarity to neighboring data points
- **Features Used**: Numerical features only (6 features)
- **Key Parameters**: 
  - k=7 neighbors
  - Euclidean distance metric
  - Feature scaling applied

### 2. **Naive Bayes (Gaussian)**
- **Purpose**: Probabilistic classification based on feature independence
- **Features Used**: All 13 features
- **Advantages**: Fast training, good baseline performance

### 3. **Support Vector Machine (SVM)**
- **Purpose**: Find optimal decision boundary using support vectors
- **Kernel**: Linear kernel
- **Features Used**: All 13 features with standardization
- **Preprocessing**: StandardScaler normalization

### 4. **Decision Tree**
- **Purpose**: Rule-based classification using feature splits
- **Features Used**: Selected features (ca, oldpeak, thalach, age, trestbps, chol)
- **Advantages**: Interpretable model with clear decision rules

### 5. **Ensemble Learning**

#### Random Forest (Bagging)
- **Purpose**: Combine multiple decision trees to reduce overfitting
- **Features Used**: All features with Min-Max scaling
- **Data Balancing**: Applied to handle class imbalance

#### Blending Model
- **Purpose**: Combine predictions from multiple models
- **Base Models**: Decision Tree + KNN
- **Meta-learner**: Logistic Regression
- **Architecture**: Two-stage ensemble approach

## 🔧 Preprocessing Pipeline

### 1. **Data Cleaning**
- Remove missing values (represented as '?')
- Binary target encoding (converting multi-class to binary)
- Data type conversion to float

### 2. **Feature Engineering**
- **Categorical Feature Removal**: Excluded categorical features for specific algorithms
- **Feature Selection**: Applied univariate selection using chi-squared test
- **Feature Scaling**: StandardScaler and MinMaxScaler normalization

### 3. **Feature Selection Techniques**
- **Univariate Selection**: SelectKBest with chi-squared test
- **Feature Importance**: Random Forest feature importance analysis
- **Correlation Analysis**: Heatmap visualization of feature correlations

## 📈 Model Evaluation

### Performance Metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate (weighted, macro, micro averages)
- **Recall**: Sensitivity (weighted, macro, micro averages)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification results

### Cross-Validation:
- **Train-Test Split**: 80-20 ratio
- **Random State**: 42 (for reproducibility)
- **Stratification**: Applied for balanced sampling

## 🎨 Visualizations

- **Correlation Heatmaps**: Feature relationship analysis
- **Feature Importance Plots**: Random Forest feature rankings
- **Confusion Matrices**: Model performance visualization
- **Class Distribution**: Target variable balance analysis

## 🚀 Getting Started

### Prerequisites
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
```

### Installation
```bash
pip install numpy pandas scikit-learn matplotlib seaborn ace_tools_open
```

### Running the Project
1. Clone the repository
2. Install required packages
3. Open `TUGAS_CLEVELAND.ipynb` in Jupyter Notebook
4. Run all cells sequentially

## 📁 Project Structure
```
Cleveland/
├── processed.cleveland.data      # Raw dataset
├── TUGAS_CLEVELAND.ipynb        # Main analysis notebook
└── README.md                    # Project documentation
```

## 🔍 Key Findings

The project demonstrates comprehensive machine learning workflow including:
- Data preprocessing and cleaning
- Feature selection and engineering
- Multiple algorithm implementation
- Model comparison and evaluation
- Ensemble learning techniques

Each algorithm provides different insights into the heart disease prediction problem, with ensemble methods typically showing improved performance through model combination.

## 👥 Contributors

This project was developed as part of a college assignment focusing on machine learning applications in healthcare.

## 📝 License

This project is for educational purposes and uses publicly available datasets.

---

**Note**: This project demonstrates various machine learning techniques for binary classification in healthcare applications. The models should not be used for actual medical diagnosis without proper validation and medical supervision.