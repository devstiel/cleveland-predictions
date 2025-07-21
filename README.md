# Cleveland Heart Disease Machine Learning Project

## 📋 Overview

This project implements multiple machine learning algorithms to predict heart disease using the Cleveland Heart Disease dataset. The goal is to build and compare various classification models to accurately identify patients at risk of heart disease based on medical attributes.

## 🎯 Objective

The primary objectives of this project are:

1. **Predictive Modeling**: Develop accurate machine learning models to predict heart disease presence based on clinical features
2. **Algorithm Comparison**: Compare the performance of different ML algorithms (KNN, Naive Bayes, SVM, Decision Tree, Random Forest, Ensemble methods)
3. **Feature Analysis**: Identify the most important features that contribute to heart disease prediction
4. **Healthcare Application**: Demonstrate the practical application of machine learning in medical diagnosis and risk assessment
5. **Model Optimization**: Implement preprocessing techniques, feature selection, and ensemble methods to improve prediction accuracy

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

## 📈 Model Evaluation & Results

### Performance Metrics:
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate (weighted, macro, micro averages)
- **Recall**: Sensitivity (weighted, macro, micro averages)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification results

### Model Performance Summary:
| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|---------|----------|
| KNN (k=7) | ~85% | High | Good | Balanced |
| Naive Bayes | ~83% | Good | High | Good |
| SVM (Linear) | ~87% | High | Good | High |
| Decision Tree | ~82% | Good | Good | Good |
| Random Forest | ~88% | High | High | High |
| Ensemble (Blending) | ~89% | High | High | Best |

### Cross-Validation:
- **Train-Test Split**: 80-20 ratio
- **Random State**: 42 (for reproducibility)
- **Stratification**: Applied for balanced sampling
- **Validation Strategy**: Cross-validation used to ensure model robustness

## 🎨 Visualizations

The project includes comprehensive data visualization:

- **Correlation Heatmaps**: Feature relationship analysis and multicollinearity detection
- **Feature Importance Plots**: Random Forest feature rankings and selection insights
- **Confusion Matrices**: Model performance visualization for each algorithm
- **Class Distribution**: Target variable balance analysis and data distribution
- **Decision Tree Visualization**: Interpretable tree structure using matplotlib
- **ROC Curves**: Model comparison using Area Under Curve (AUC) metrics
- **Learning Curves**: Training vs validation performance analysis

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
1. **Clone the repository**:
   ```bash
   git clone https://github.com/devstiel/cleveland-predictions.git
   cd cleveland-predictions
   ```

2. **Install required packages**:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Open and run the analysis**:
   - Open `TUGAS_CLEVELAND.ipynb` in Jupyter Notebook
   - Run all cells sequentially to reproduce the analysis
   - View generated visualizations and model results

5. **Explore the results**:
   - Check generated PNG files for visualizations
   - Review model performance metrics in the notebook
   - Analyze feature importance and decision boundaries

## 📁 Project Structure
```
cleveland-predictions/
├── processed.cleveland.data      # Raw Cleveland heart disease dataset
├── TUGAS_CLEVELAND.ipynb        # Main analysis notebook with all implementations
├── heart_disease_tree.png       # Decision tree visualization
├── heart_disease_tree_matplotlib.png  # Alternative tree visualization
├── supervised_ml_comparison.png # Model performance comparison chart
├── README.md                    # Comprehensive project documentation
└── requirements.txt             # Python dependencies (if available)
```

## 🔍 Key Findings & Insights

The comprehensive analysis reveals several important insights:

### Algorithm Performance:
1. **Ensemble Methods** (Random Forest, Blending) achieve the highest accuracy (~88-89%)
2. **SVM with Linear Kernel** shows excellent performance (~87%) with good generalization
3. **KNN** demonstrates solid performance (~85%) with proper feature scaling
4. **Decision Trees** provide excellent interpretability but lower accuracy (~82%)

### Feature Importance:
- **ca** (number of major vessels): Most significant predictor
- **oldpeak** (ST depression): Strong indicator of heart disease
- **thalach** (max heart rate): Important cardiovascular marker
- **age** and **sex**: Demographic factors with moderate importance

### Technical Insights:
- Feature scaling significantly improves KNN and SVM performance
- Ensemble methods effectively reduce overfitting
- Proper data preprocessing is crucial for model performance
- Feature selection helps in reducing noise and improving interpretability

### Clinical Relevance:
- The models identify key cardiovascular risk factors
- High precision reduces false positive diagnoses
- High recall ensures most at-risk patients are identified
- Ensemble approach provides more reliable predictions for clinical decision support

## 👥 Contributors

This project was developed as part of a machine learning course assignment, demonstrating practical applications of ML algorithms in healthcare.

**Project Focus Areas**:
- Healthcare Data Science
- Supervised Learning Algorithms
- Model Evaluation and Comparison
- Feature Engineering and Selection
- Ensemble Learning Methods

## 🏆 Project Achievements

- ✅ Successfully implemented 6 different ML algorithms
- ✅ Achieved 89% accuracy with ensemble methods
- ✅ Created comprehensive visualizations for model interpretation
- ✅ Developed robust preprocessing pipeline
- ✅ Demonstrated practical application of ML in healthcare
- ✅ Provided detailed documentation and reproducible code

## 🔗 References

- **Dataset Source**: UCI Machine Learning Repository - Cleveland Heart Disease Dataset
- **Algorithms**: Scikit-learn machine learning library
- **Visualization**: Matplotlib and Seaborn libraries
- **Research Papers**: Various studies on ML applications in cardiovascular disease prediction

## 📞 Contact

For questions about this project or collaboration opportunities, please feel free to reach out through GitHub.

## 📝 License

This project is for educational purposes and uses publicly available datasets. All code is available under MIT License.

---

**⚠️ Medical Disclaimer**: This project is for educational and research purposes only. The models and predictions should not be used for actual medical diagnosis without proper clinical validation and medical supervision. Always consult qualified healthcare professionals for medical advice.