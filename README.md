# Human Activity Recognition Using Machine Learning

A machine learning project that classifies human activities based on sensor data from smartphones and wearable devices. The model achieves over 96% accuracy in distinguishing between six different activities using ensemble learning techniques.

## Project Overview

This project tackles the Human Activity Recognition (HAR) problem using data from accelerometers, gyroscopes, and magnetometers. These sensors, commonly found in smartphones and wearable devices, capture motion and orientation patterns that correspond to different human activities.

### Activities Classified

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

## Dataset

The dataset contains sensor readings with labeled activities. Each sample includes multiple numeric features extracted from sensor measurements at regular time intervals. The data is well-balanced across all activity classes with no missing values.

## Methodology

### 1. Exploratory Data Analysis
- Analyzed statistical distributions of features
- Visualized activity class distributions
- Examined feature correlations
- Confirmed data quality with no missing values

### 2. Feature Engineering
- **Data Splitting**: Created 80-20 train-validation split
- **Outlier Detection**: Identified outliers using Z-score analysis
- **Variance Thresholding**: Removed low-variance features (threshold = 0.01)
- **Dimensionality Reduction**: Applied PCA to retain 95% variance
- **Standardization**: Scaled features using StandardScaler

### 3. Model Development
I experimented with multiple algorithms and combined them using ensemble methods:

**Individual Models:**
- Random Forest Classifier
- Logistic Regression
- Support Vector Machine (SVM)

**Hyperparameter Optimization:**
- Used GridSearchCV with 3-fold cross-validation
- Optimized for weighted F1-score
- Tested various combinations of hyperparameters

**Ensemble Strategy:**
- Implemented soft voting ensemble
- Combined RF, LR, and SVM with weights 2:1:2
- Leveraged strengths of each individual model

### 4. Key Findings

**Random Forest:**
- 300 estimators with max depth of 30
- Used sqrt for max_features to maintain tree diversity
- Balanced accuracy and computational efficiency

**Logistic Regression:**
- Regularization parameter C in range 1-5
- LBFGS solver with multinomial setting
- Worked effectively with standardized features

**SVM:**
- RBF kernel for non-linear decision boundaries
- C=5 with gamma=scale
- Excellent at capturing complex sensor patterns

**Ensemble Results:**
- Weighted voting improved overall performance
- Achieved 96%+ F1-score on validation set
- Strong generalization to unseen test data

## Results

The final ensemble model achieved:
- **Accuracy**: 96%+
- **Precision**: 96%+ (weighted)
- **Recall**: 96%+ (weighted)
- **F1-Score**: 96%+ (weighted)

The confusion matrices show excellent separation between activity classes with minimal misclassifications, particularly between similar activities like sitting and standing.

## Technologies Used

- **Python 3.x**
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **scikit-learn**: Machine learning algorithms and tools
- **Matplotlib & Seaborn**: Data visualization
- **SciPy**: Statistical analysis

## Project Structure

```
.
├── human_activity_classification.ipynb   # Main analysis notebook
├── train.csv                             # Training dataset
├── test.csv                              # Test dataset
├── submission.csv                        # Model predictions
└── README.md                             # Project documentation
```

## Usage

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn scipy
   ```
3. Open the Jupyter notebook:
   ```bash
   jupyter notebook human_activity_classification.ipynb
   ```
4. Run the cells sequentially to reproduce the analysis

## Model Pipeline

1. Load training and test data
2. Perform exploratory data analysis
3. Apply feature engineering techniques
4. Train individual models with hyperparameter tuning
5. Create voting ensemble
6. Evaluate on validation set
7. Generate predictions for test set

## Future Improvements

- Experiment with deep learning approaches (LSTM, CNN)
- Implement real-time activity recognition
- Add more activity classes
- Explore feature importance analysis
- Test on different sensor configurations

## License

This project is available for educational and research purposes.

## Acknowledgments

This project was developed as part of a machine learning course focusing on classification problems and ensemble methods.