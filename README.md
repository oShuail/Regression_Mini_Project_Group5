# Insurance Premium Prediction Using Decision Tree Regression

Group 5 mini project: predicting a customer's yearly insurance expenses from their personal information using a **Decision Tree Regressor**.

The notebook covers the full machine learning workflow: exploratory data analysis (EDA), preprocessing, model training and evaluation.

---

## Dataset

**[Insurance Premium Prediction](https://www.kaggle.com/datasets/noordeen/insurance-premium-prediction)** (Kaggle): 1,338 customers, 6 features plus the target, no missing values.

| Column | Type | Description |
|---|---|---|
| `age` | Numeric | Age of the customer (18–64) |
| `sex` | Categorical | Female / male |
| `bmi` | Numeric | Body Mass Index (16–53) |
| `children` | Numeric | Number of children covered (0–5) |
| `smoker` | Categorical | Yes / no |
| `region` | Categorical | Northeast, northwest, southeast, southwest |
| `expenses` | Numeric (**target**) | Yearly insurance expenses ($1,122 – $63,770) |

---

## Workflow

### 1. Exploratory Data Analysis
- **Target:** `expenses` is strongly right-skewed (mean $13,270 vs. median $9,382). Most customers have low costs, while a few have very high costs.
- **Smoking status is the strongest factor:** smokers have much higher expenses than non-smokers and form a separate, higher band in every plot.
- **Age** has the strongest numeric correlation with expenses (0.30), followed by **BMI** (0.20). Expenses rise with age for both smokers and non-smokers.
- **BMI matters mostly for smokers:** smokers with a higher BMI have much higher expenses, while for non-smokers the relationship is weak.
- `children`, `sex` and `region` show little effect on expenses.
- The classes are balanced for `sex` and `region`, but imbalanced for `smoker` (about 80% non-smokers).
- There's no multicollinearity: all pairwise correlations between predictors are low.

### 2. Data Preprocessing
- Checked for missing values (none) and removed **1 duplicate row**, leaving 1,337 rows.
- Encoded `sex` (female = 0, male = 1) and `smoker` (no = 0, yes = 1).
- One-hot encoded `region` into 4 columns.
- No scaling was needed, because decision trees split on thresholds and aren't affected by feature scale.

### 3. Model Training
- Train/test split: **80% / 20%** (`random_state=42`).
- Model: `DecisionTreeRegressor` with limits to reduce overfitting:

| Hyperparameter | Value |
|---|---|
| `max_depth` | 5 |
| `min_samples_split` | 10 |
| `min_samples_leaf` | 5 |

### 4. Model Evaluation (test set)

| Metric | Value | Meaning |
|---|---|---|
| **R²** | **0.895** | The model explains about 89.5% of the variation in expenses |
| **MAE** | $2,609 | Predictions are off by about $2,609 on average |
| **RMSE** | $4,390 | Much higher than MAE, so a few predictions have large errors (mostly high-cost customers) |
| **MSE** | 19,268,545 | In squared dollars; mainly useful for comparing models |

---

## Key Findings

- The Decision Tree performs well, with an **R² of 0.895** on unseen data.
- **Smoking status** is the most important factor affecting insurance expenses, followed by age and BMI.
- Limiting the tree's depth and leaf size kept the model simple and reduced the risk of overfitting.

## Limitations

- RMSE is much higher than MAE, so the model still makes some large errors, mostly on high-cost customers.
- A single decision tree gives step-like predictions, since every customer in the same leaf gets the same predicted value.

## Possible Improvements

- Tune the hyperparameters with `GridSearchCV`.
- Try a **Random Forest**, which combines many trees and usually gives more accurate and stable predictions.

---

## How to Run

1. Open `Mini_Project_Group5.ipynb` in [Google Colab](https://colab.research.google.com/) or Jupyter.
2. Install the requirements (already available on Colab):
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
   ```
3. Run all cells. The dataset downloads automatically with `kagglehub`:
   ```python
   path = kagglehub.dataset_download("noordeen/insurance-premium-prediction")
   ```

---

## Repository Structure

```
├── Mini_Project_Group5.ipynb   # Full notebook: EDA, preprocessing, training, evaluation
└── README.md
```

---

## Team (Group 5)

| Member | Part |
|---|---|
| [Raghad Alasmari](https://github.com/Raghad0za) | Dataset exploration (EDA) |
| Ruba | Data preprocessing |
| [Daniyah Aleid](https://github.com/DAleid) | Model training |
| [Omar Alshuail](https://github.com/oShuail) | Model evaluation |
