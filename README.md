# California-Housing-Model-
Exploration, preprocessing and model training using California housing dataset
# Predicting California Housing Prices with Regularized Regression


## What This Project Does

Housing prices depend on a mix of income, location, and property characteristics — but which factors actually matter, and can regularization sharpen a model's predictions? This project tackles that question using the **California Housing dataset**, comparing four regression techniques side by side:

- Ordinary **Linear Regression**
- **Ridge Regression** (shrinks coefficients using an L2 penalty)
- **Lasso Regression** (can eliminate weak predictors entirely via an L1 penalty)
- **Elastic Net** (a blend of both penalties)

We also trained a **Stochastic Gradient Descent (SGD) Regressor** as an additional benchmark.

## Dataset Snapshot

Pulled directly from `sklearn.datasets.fetch_california_housing`:

- **20,640 district-level records** across California
- **Target:** `MedHouseVal` (median home value, in $100,000s)
- **8 numeric predictors:** income (`MedInc`), housing age, average rooms/bedrooms, population, occupancy, and geographic coordinates (`Latitude`, `Longitude`)
- No categorical variables were present, so preprocessing focused entirely on scaling rather than encoding.

## How the Analysis Was Built

**Step 1 — Explore the data.** Checked structure, missing values, duplicates, summary stats, and outliers using the IQR method. Visualized the target's distribution, a correlation heatmap, and scatterplots against key predictors.

**Step 2 — Preprocess carefully.** Split the data 80/20 *before* scaling, then fit `StandardScaler` only on the training set — this avoids leaking test-set information into training, a common pitfall in regression workflows.

**Step 3 — Establish a baseline.** Trained a plain Linear Regression model on all 8 features, then validated it further using 5-fold cross-validation to check for overfitting.

**Step 4 — Tune the regularized models.** Built Ridge, Lasso, and Elastic Net as scikit-learn `Pipeline`s (scaler + model), then searched for each model's best hyperparameters using `GridSearchCV` across a range of alpha values (and l1_ratio for Elastic Net).

**Step 5 — Compare everything.** Evaluated all four models on the same test set using MSE, RMSE, MAE, and R², and examined each model's coefficients side by side to see how regularization reshaped them.

## What We Found

| Model | Best Alpha | Best L1 Ratio | Test MSE | Test RMSE | Test MAE | Test R² |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Linear Regression | — | — | 0.5559 | 0.7456 | 0.5332 | 0.5758 |
| Ridge | 0.0001 | — | 0.5559 | 0.7455 | 0.5332 | 0.5758 |
| Lasso | 0.001 | — | **0.5545** | **0.7446** | 0.5331 | **0.5769** |
| Elastic Net | 0.001 | 0.9 | 0.5546 | 0.7447 | **0.5331** | 0.5768 |

**Takeaways:**

- **Lasso edged out the competition**, though barely — every model landed within a hair of each other on test performance.
- Regularization didn't move the needle much here. That's actually a meaningful finding: it suggests the original Linear Regression model wasn't badly overfit to begin with, so there wasn't much for a penalty term to correct.
- Interestingly, **no coefficients were driven to zero** by Lasso or Elastic Net, even though that's typically their signature move. With such small optimal alphas, the models effectively decided every one of the 8 features was pulling its weight.
- **Location dominates.** Latitude and Longitude consistently had the largest coefficients across every model — a reminder that in California, where you are matters more than almost anything else for home value.

## Built With

- Python · pandas · NumPy · Matplotlib
- scikit-learn — `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SGDRegressor`, `Pipeline`, `GridSearchCV`, `KFold`


## Collaborators

Medicine Group1, TechRise Cohort 3 — AI & ML Track

ELECHI CHINENYE MERCY 
GitHub: https://github.com/chinenyemercy5228-ship-it
linkedin: www.linkedin.com/in/elechi-chinenye-freelancer

ONWUKWE IHEANYI EMMANUEL
GitHub: https://github.com/iheanyi-dev

CHUKWUEMEKA VICTOR CHUKWUEMEKA
Github: https://github.com/VCT2008-bot
Linkedin: www.linkedin.com/in/victor-chukwuemeka-204738419

OKPARA KAMSY SOPHIA
GitHub: https://github.com/Sophiak499

