# Data Preprocessing, Exploratory Data Analysis (EDA), and Regression Modeling

MSc Applied AI & ML coursework (Assignment 1): preparing the California Housing dataset and comparing five regression models that predict median house value (`MedHouseVal`).

## Results

Evaluated on a held-out 20% test set (`random_state=42`).

| Model | MSE | R² |
|---|---|---|
| **Random Forest** (100 trees) | **0.36** | **0.72** |
| Support Vector Regression (RBF, C=100, γ=0.1, ε=0.1) | 0.44 | 0.66 |
| Decision Tree (max depth 10) | 0.50 | 0.62 |
| Polynomial Regression (degree 2) | 0.55 | 0.58 |
| Linear Regression | 0.65 | 0.50 |

Random Forest performed best: averaging many decision trees reduces variance and captures non-linear patterns that the linear model misses. The notebook has the full analysis of each model.

## What the notebook does

1. **Load and inspect** the dataset with `fetch_california_housing` (20,640 rows).
2. **Handle missing data:** introduce missing `MedInc` values in rows 0–20, then impute them with the median.
3. **Encode:** bin `HouseAge` into Young, Middle-Aged and Old, then one-hot encode with `drop_first=True`.
4. **Scale** the numerical features with `StandardScaler`.
5. **Explore:** summary statistics, histograms, and a scatter plot of income against house value.
6. **Select features:** drop one feature from each pair with absolute correlation above 0.8, which removes `AveBedrms` and `Longitude`.
7. **Save and split:** write the processed data to CSV and split it 80/20 into train and test sets.
8. **Train and evaluate** the five regression models using MSE and R².

## Files

| File | Description |
|---|---|
| [`Assignment1_template.ipynb`](Assignment1_template.ipynb) | Completed notebook with code, plots and analysis |
| [`preprocessed_housing_data.csv`](preprocessed_housing_data.csv) | Processed dataset produced by the notebook |

## Running it

The notebook was run on Python 3.12.

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook Assignment1_template.ipynb
```

scikit-learn downloads the dataset the first time it runs, so you need an internet connection. The SVR step can take a few minutes to train.
