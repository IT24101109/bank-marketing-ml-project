# Bank Marketing Subscription Prediction

An end-to-end machine learning project that explores whether historical bank-campaign data can help rank customers by their likelihood of subscribing to a term deposit.

## Project overview

The notebook uses the UCI Bank Marketing dataset and covers data quality checks, exploratory analysis, feature preparation, model comparison, time-aware cross-validation, tuning, and final holdout evaluation.

**Business question:** Can information available before a customer call help prioritize outreach to customers who may subscribe to a term deposit?

The target is `y` (`yes` or `no`). This is a binary classification problem with an imbalanced target, so the notebook reports metrics beyond accuracy.

## Notebook

Open [`bank_marketing_subscription_prediction.ipynb`](bank_marketing_subscription_prediction.ipynb) in GitHub, Jupyter Notebook, or JupyterLab. The notebook is organized with explanations before each major step and is intended to run from top to bottom.

It includes:

- Dataset loading directly from the UCI archive; no manual data download is needed.
- Dataset overview, missing-value and duplicate checks, and exploratory plots.
- Feature engineering for previous-contact history.
- Exclusion of `duration`, which is known only during/after a call and would leak information for a pre-call prediction use case.
- Preprocessing pipelines for numeric and categorical features.
- A majority-class baseline, logistic regression, and random forest.
- Expanding-window 5-fold cross-validation and logistic-regression hyperparameter tuning.
- A final evaluation on the latest 20% of source rows, with ROC-AUC, average precision, precision, recall, F1, balanced accuracy, and a confusion matrix.
- Logistic-regression coefficient interpretation and a concise findings section.

## Run it locally

1. Install Python 3.10 or later.
2. Install the packages used by the notebook:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Start Jupyter in the folder containing the notebook:

   ```bash
   jupyter notebook
   ```

4. Open `bank_marketing_subscription_prediction.ipynb` and choose **Run All**.

An internet connection is needed when the notebook downloads the dataset from UCI. The analysis uses a fixed random seed (`42`) for randomized estimators.

## Validation approach and limitations

UCI documents `bank-additional-full.csv` as ordered by date, but the file does not provide an exact contact date. The notebook therefore uses source row order as an approximate time index: the first 80% is used for development and the last 20% is held out; `TimeSeriesSplit` creates expanding-window folds within the development period. This is more appropriate for forward prediction than a shuffled split, but it is not a true date-based backtest.

The project uses historical data from 2008–2010. Its metrics should not be treated as evidence of current campaign performance. Validate on recent, genuinely later-dated campaigns before operational use. The default 0.50 threshold is illustrative; select a business threshold using campaign capacity and the costs of missed subscriptions and additional calls. The current-campaign contact count is used under the assumption that it is known at the scoring time.

## Dataset and attribution

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. [Dataset page](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing) · [DOI: 10.24432/C5K306](https://doi.org/10.24432/C5K306). Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Results

Run the notebook to generate the current cross-validation and latest-period test metrics. Keep this section updated with the results from a complete run before using the repository as a portfolio project; the notebook's source-order split may produce different estimates from a random split.

## Suggested repository structure

```text
bank-marketing-ml-project/
├── README.md
└── bank_marketing_subscription_prediction.ipynb
```
