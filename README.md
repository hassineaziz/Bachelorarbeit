# Cost-Sensitive Fraud Detection — Jupyter Notebook

This notebook contains the complete code for my Bachelor's thesis,
*Cost-Sensitive Fraud Detection in Financial Transactions: A Comparative
Analysis of XGBoost and Neural Networks to Reduce False Negatives.*

It runs the full pipeline on two credit-card fraud datasets: data
exploration, preprocessing, baseline models, the cost-sensitive
experiments, and the final evaluation.

- **Notebook:** `bachelorarbeit.ipynb`
- **Author:** Aziz Hassine
- **Language:** Python 3.12

## Datasets

The notebook uses two public datasets. Neither is included in this
submission because of their size; both are on Kaggle.

| Name                              | Rows    | Fraud rate | Source                                                        |
|-----------------------------------|---------|------------|---------------------------------------------------------------|
| IEEE-CIS Fraud Detection          | 590,540 | 3.5%       | https://www.kaggle.com/competitions/ieee-fraud-detection/data |
| Credit Card Fraud Detection (ULB) | 284,807 | 0.172%     | https://www.kaggle.com/mlg-ulb/creditcardfraud                |

The IEEE-CIS data comes in two files that are joined on `TransactionID`:
`train_transaction.csv` and `train_identity.csv`. The ULB data is a single
file (`creditcard.csv`).

## What the notebook does

1. **Setup and data loading** — reads the three CSV files.
2. **Dataset exploration** — class imbalance, duplicate rows, transaction amount distributions, missing-value overview, and the relationship between individual features and the fraud label.
3. **Preprocessing**
   - an informative-missingness check that keeps a column's missingness as a flag when it predicts fraud, instead of dropping the column outright;
   - constant fill for categorical missing values, one-hot encoding of low-cardinality categoricals;
   - a stratified three-way split (train / validation / test) done *before* any fitting, so nothing leaks from the test set;
   - numeric imputation with training-set medians and standardization, both fit on the training set only.
4. **Cost-sensitive framework** — the Bahnsen et al. (2013) cost model, the evaluation metrics (recall, precision, PR-AUC, false-negative and false-positive counts, cost, savings), the three decision rules (default 0.5 threshold, global cost-optimal threshold `t*`, and the amount-dependent threshold `p* = Ca / amount`), and isotonic probability calibration for the reweighted models.
5. **Models** — for each dataset, XGBoost and a neural network are each trained in three variants: baseline, class-weighted, and amount-weighted. XGBoost is tuned once per dataset with a randomized search scored on PR-AUC.
6. **Results** — every combination of (variant × cost × decision rule) is evaluated on the held-out test set, thresholds are chosen on validation only, and the full table is saved to `cost_sensitive_results.csv`. The final cells show the best configuration per model, a baseline vs. cost-sensitive comparison, the sensitivity analysis over the administrative cost, and before/after confusion matrices.

## How to run it

The notebook was written and run on **Kaggle** with a GPU, which is the
simplest way to reproduce it.

### On Kaggle (recommended)

1. Create a new Kaggle Notebook and upload `bachelorarbeit.ipynb`.
2. Add the two datasets above to the notebook (Add Input).
3. Check that `base_path` in the second code cell points to the folder where the CSV files are mounted. The first cell prints every file under `/kaggle/input`, which makes the correct path easy to find.
4. Turn the accelerator on (GPU) and run all cells from top to bottom.

### Locally

1. Install the dependencies:

   ```
   pip install pandas numpy scikit-learn xgboost tensorflow
   ```
2. Download the three CSV files and update `base_path` to point to them.
3. If you do not have a GPU, set `DEVICE = 'cpu'` in the XGBoost cell (Section 5.2). The neural network runs on CPU as well, just more slowly.
4. Run all cells from top to bottom.

## Notes

- **Run order matters.** The cells are meant to be run once, in order, from top to bottom. Several preprocessing cells change the dataframes in place, so re-running a single cell out of order can give wrong results; restart the kernel and run all if in doubt.
- **Reproducibility.** All random seeds (data splitting, XGBoost sampling, and neural-network initialisation) are fixed to 42.
- **Output.** The main artifact produced is `cost_sensitive_results.csv`, which holds every result reported in the thesis.

## Libraries used

pandas, NumPy, scikit-learn, XGBoost, and TensorFlow/Keras.