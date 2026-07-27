# MSCS_634_Lab_4 — Regression Techniques and Regularization

**Name:** Richin Swaroop Dasari
**Course:** 2026 Summer - Advanced Big Data and Data Mining (MSCS-634-B01) - Second Bi-term
**Assignment:** Lab 4: Regression Analysis with Regularization Techniques

## Purpose

For this lab I worked with the **Diabetes Dataset** from `sklearn.datasets`, which has a set of health measurements used to predict how much a patient's disease progressed over a year. The goal was to:

- Build Simple Linear Regression, Multiple Regression, and Polynomial Regression models.
- Apply Ridge and Lasso Regression and see how regularization helps with overfitting.
- Evaluate everything using MAE, MSE, RMSE, and R².
- Visualize the predictions so I could actually see how each model was performing, not just look at numbers.

## Contents

- `MSCS_634_Lab_4.ipynb` — the notebook with all the data exploration, modeling, evaluation, and plots.
- `README.md` — this file.

## Key Insights

- **Multiple Regression did noticeably better than Simple Linear Regression.** Using all 10 features instead of just BMI pushed R² up quite a bit — makes sense, since disease progression depends on a mix of things like BMI, blood pressure, and a few of the serum measurements, not just one variable.
- **Polynomial Regression made the overfitting tradeoff really obvious.** As I bumped the degree up from 1 to 4, training R² kept going up (or stayed high) while test R² dropped off a cliff at the higher degrees. That's the model fitting noise instead of the real pattern, which isn't too surprising given the dataset only has 442 samples.
- **Ridge and Lasso both helped rein in the overfitting** from the expanded polynomial features. Both generalized better to the test set than the plain polynomial model did. Lasso went further and zeroed out a bunch of coefficients on its own — basically doing feature selection for me — while Ridge shrank everything smoothly without eliminating anything.
- **Alpha controls the bias-variance tradeoff, and you can really see it in the results.** Too small an alpha and the model behaves like plain least squares (more overfitting risk); too large and it shrinks the coefficients so much it starts underfitting. For both Ridge and Lasso, the best test R² landed somewhere in the middle, not at either extreme.
- **BMI and the `s5` serum measurement showed up as the strongest predictors** across pretty much every model I tried, which was a nice consistency check that the models were picking up on something real.

## Challenges and Decisions

- **Picking a feature for Simple Linear Regression:** I went with BMI since it had the highest correlation with the target out of all the features.
- **Scaling before Ridge/Lasso:** Since Ridge and Lasso penalize coefficients based on their size, I standardized the degree-2 polynomial features with `StandardScaler` first — otherwise features on a bigger scale would get penalized more just because of their units, not because they actually matter less.
- **Lasso not converging at very low alpha:** At the smallest alpha value I tried, Lasso's solver threw a convergence warning. I bumped `max_iter` up to 10,000 to help, but honestly at that low an alpha the model is basically behaving like ordinary least squares anyway, so it didn't change my overall conclusions.
- **Why degree 2 for Ridge/Lasso:** I used the degree-2 polynomial feature set as the base for regularization since that's where I'd already seen overfitting start to creep in with plain polynomial regression, so it was a good test case for seeing whether Ridge/Lasso could actually fix it.

## How to Run

1. Open `MSCS_634_Lab_4.ipynb` in Jupyter Notebook or JupyterLab.
2. Run all cells top to bottom (`Kernel > Restart & Run All`).
3. Needs: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.
