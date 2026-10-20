# Practical Lab 2 - Diabetes Progression Prediction Using Machine Learning

**Student:** Zeynep Ozdemir  
**Student ID:** 9045142  
**Course:** CSCN8010 - Foundation Machine Learning Framework

---

## Project Summary

This project explores diabetes disease progression prediction using the Scikit-Learn Diabetes dataset.

The objective is to develop and compare machine learning models that predict the quantitative diabetes disease progression score one year after baseline. A separate classification approach is also evaluated to identify patients with higher progression scores.

The project implements:

- Univariate Polynomial Regression
- Multivariate Polynomial Regression
- Multiple Linear Regression
- Decision Tree Regression
- k-Nearest Neighbors (kNN) Regression
- Logistic Regression Classification
- 5-Fold Cross-Validation

Models are evaluated using appropriate regression and classification performance metrics to compare prediction accuracy, generalization, and model complexity.

## Dataset

The project uses the Scikit-Learn Diabetes dataset, which contains:

- **442 observations**
- **10 numerical input features**
- **1 continuous target variable:** Diabetes disease progression one year after baseline

The dataset includes the following features:

- Age
- Sex
- Body Mass Index (BMI)
- Blood Pressure (BP)
- Six blood serum measurements (s1–s6)

The input features are scaled using the default Scikit-Learn dataset settings.

Exploratory Data Analysis (EDA) includes:

- Descriptive statistics
- Missing-value and duplicate checks
- Histograms
- Scatter plots
- Correlation matrix
- Feature relationship analysis

No additional data cleaning was required based on the initial data checks.

The dataset is divided into:

| Dataset | Percentage |
|---|---:|
| Training | 75% |
| Validation | 10% |
| Testing | 15% |

The test dataset is reserved for final model evaluation.

**Dataset Documentation:**  
https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset

## Regression Models and Evaluation

### Part 2 - Univariate Polynomial Regression

Six polynomial regression models are trained using BMI as the only independent variable.

The polynomial degrees evaluated are:

- Degree 0
- Degree 1
- Degree 2
- Degree 3
- Degree 4
- Degree 5

Each model is evaluated using:

- R-Squared (R²)
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

Training and validation results are compared to identify the best-performing model.

The notebook also includes:

- Visualization of training, validation, and test data
- Fitted polynomial regression curve
- Mathematical equation of the selected model
- Prediction for a selected BMI value
- Comparison of trainable parameter counts
- Discussion of model complexity and overfitting

**Selected Model:** Polynomial Regression - Degree 3

### Part 3 - Multivariate Regression

The multivariate regression analysis uses four selected input features:

- BMI
- Blood Pressure (BP)
- S5
- S3

Seven regression models are evaluated:

| Model | Configuration |
|---|---|
| Multiple Linear Regression | Baseline |
| Polynomial Regression | Degree 2 |
| Polynomial Regression | Degree 3 |
| Decision Tree Regression | Max Depth = 3 |
| Decision Tree Regression | Max Depth = 5 |
| k-Nearest Neighbors | k = 5 |
| k-Nearest Neighbors | k = 15 |

The models are compared using training and validation R², MAE, and MAPE.

The model with the lowest validation MAE is selected for final testing.

**Selected Model:** Multivariate Polynomial Regression - Degree 2

## Logistic Regression Classification

A separate classification experiment converts the continuous disease progression target into two categories:

- **Class 0:** Low progression
- **Class 1:** High progression

The threshold is determined using the median progression score from the training dataset.

Two logistic regression classifiers are evaluated:

- Logistic Regression (C = 0.1)
- Logistic Regression (C = 1.0)

Classification performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

The selected model is based on validation recall because identifying high-progression cases is important in a screening context.

**Selected Classifier:** Logistic Regression (C = 0.1)

## Results

### Test Results - Regression

| Model | Test R² | Test MAE | Test MAPE |
|---|---:|---:|---:|
| BMI Polynomial Regression (Degree 3) | 0.207 | 54.63 | 48.54% |
| Multivariate Polynomial Regression (Degree 2) | 0.438 | 44.20 | 38.71% |

The multivariate polynomial regression model achieves better test performance than the BMI-only model.

Using multiple patient features improves prediction accuracy and reduces average prediction errors.

However, prediction errors remain substantial, indicating that the models have limitations when estimating individual patient progression scores.

### Test Results - Classification

| Metric | Logistic Regression (C = 0.1) |
|---|---:|
| Accuracy | 76.12% |
| Precision | 71.88% |
| Recall | 76.67% |
| F1 Score | 74.19% |

The selected logistic regression classifier missed **7 high-progression cases** in the test dataset.

These false negatives are important because patients with potentially higher progression could be classified as lower risk.

## 5-Fold Cross-Validation

To evaluate model stability and generalization, 5-fold cross-validation is performed on the selected multivariate polynomial regression model (Degree 2).

The training dataset is divided into five folds. The model is trained on four folds and evaluated on the remaining fold.

This process is repeated five times, with each fold used once for evaluation.

The test dataset is not used during cross-validation.

### Cross-Validation Results

| Metric | Result |
|---|---:|
| Mean R² | 0.435 |
| Mean MAE | 47.74 |
| Mean MAPE | 41.59% |
| R² Standard Deviation | 0.035 |

The relatively small R² standard deviation suggests consistent model performance across the five folds.

The cross-validation mean R² is also close to the final test R² of 0.438.

Since the selected model was identified using validation data before cross-validation, these results are interpreted as a stability check rather than an unbiased comparison of all candidate models.

## Notebook Visualizations

The notebook includes visualizations to support data exploration and model evaluation:

- Feature distribution histograms
- Scatter plots of input features and diabetes progression
- Feature correlation matrix
- Regression performance comparison tables
- Polynomial regression curve with training, validation, and test points
- Model complexity and prediction error analysis
- Classification confusion matrix

These visualizations help explain feature relationships, model behavior, prediction accuracy, and potential overfitting.

All figures are generated directly inside the Jupyter Notebook.

## Project Structure

```text
diabetes-progression-ml-lab/
│
├── .venv/                         # Local Python virtual environment
│
├── data/                          # Data directory
│
├── documents/                     # Project documentation
│
├── notebooks/
│   └── diabetes_progression_analysis.ipynb
│
├── src/
│   ├── main.py
│   └── regression_metrics.py      # Reusable regression evaluation function
│
├── .gitignore
├── README.md
└── requirements.txt
```

The dataset is loaded directly from Scikit-Learn, so no separate dataset download is necessary.

## Setup Instructions

### 1. Obtain the Project

Clone the GitHub repository or download the project folder.

Open a terminal in the project root directory:

```bash
cd diabetes-progression-ml-lab
```

### 2. Create a Virtual Environment

Create a Python virtual environment:

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

Install the required Python packages:

```bash
python -m pip install -r requirements.txt
```

The main dependencies include:

- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

### 5. Open the Jupyter Notebook

Open the project folder in Visual Studio Code.

Select the Python interpreter associated with the virtual environment.

Open:

```text
notebooks/diabetes_progression_analysis.ipynb
```

Run all notebook cells from beginning to end.

The notebook imports the custom regression evaluation function using:

```python
from src.regression_metrics import regression_scores
```

The `src` folder must remain in the project root. The notebook setup adds the project root to the Python import path.

### 6. Execute the Machine Learning Analysis

Running the notebook performs the following operations:

1. Loads the diabetes dataset.
2. Conducts Exploratory Data Analysis.
3. Checks data quality.
4. Splits the dataset into training, validation, and test sets.
5. Trains univariate polynomial regression models.
6. Evaluates and compares regression performance.
7. Trains multivariate regression models.
8. Evaluates logistic regression classifiers.
9. Performs 5-fold cross-validation.
10. Displays results, visualizations, and conclusions.

No PostgreSQL database, streaming service, or external data configuration is required.

## Main Technologies

- Python
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook
- Visual Studio Code

## Model Limitations

The dataset contains only 442 observations, which limits the generalizability of the models.

Although the multivariate polynomial regression model performs better than the univariate model, it still produces meaningful prediction errors.

The regression models may underestimate high progression scores and overestimate lower progression scores.

The logistic regression classification threshold is based on the training-set median rather than a medically validated risk threshold.

False negatives in medical screening can be particularly important because high-progression patients may not be identified correctly.

The models are developed for educational purposes and should not be used independently for clinical diagnosis or screening.

Future improvements could include:

- Additional clinical data
- Feature selection and engineering
- Hyperparameter optimization
- More extensive cross-validation
- External dataset validation
- Improved sensitivity to high-progression cases

## AI Assistance

ChatGPT by OpenAI was used to assist with debugging, code organization, evaluation explanations, and documentation.

The student is responsible for reviewing, executing, testing, and verifying the final project before submission.