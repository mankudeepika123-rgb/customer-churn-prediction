# Customer Churn Prediction

Predicting which telecom customers are likely to stop using the service, using classification models and exploratory data analysis.

## Overview
Customer churn (customers leaving a service) is a major cost for subscription-based businesses. This project analyzes customer data to find the main drivers of churn and builds machine learning models to predict which customers are at risk, so a business can act before they leave.

## Dataset
- **Telco Customer Churn** dataset (7,043 customers, 21 columns)
- Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Each row is one customer, with demographic info, account info (contract, payment method, tenure), services subscribed (internet, phone, streaming, etc.), and whether they churned.

## Tools Used
- Python, Pandas, NumPy
- Matplotlib, Seaborn (visualization)
- Scikit-learn (Logistic Regression, Random Forest, preprocessing, metrics)
- XGBoost
- imbalanced-learn (SMOTE)

## Steps Followed

### 1. Data Cleaning
```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna().drop("customerID", axis=1)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
```

### 2. Exploratory Data Analysis (EDA)
Studied churn rate by contract type, internet service, payment method, tenure, and monthly charges.

![Churn count](images/01_churn_count.png)
![Churn by category](images/02_churn_by_category.png)
![Churn by tenure](images/03_churn_by_tenure.png)
![Correlation heatmap](images/04_correlation_heatmap.png)

**Key findings:**
- Customers on **month-to-month contracts** churn far more than those on one- or two-year contracts - the single strongest driver of churn.
- **Fiber optic** internet customers churn more than DSL customers.
- Customers **without online security or tech support** are more likely to churn.
- Customers who pay by **electronic check** churn more than those on other payment methods.
- **New customers (first 12 months)** are the highest-risk group for churn.

### 3. Preprocessing
```python
# Merge redundant categories, one-hot encode
df = df.replace({"No internet service": "No", "No phone service": "No"})
X = pd.get_dummies(df.drop("Churn", axis=1), drop_first=True, dtype=int)
y = df["Churn"]

# 80/20 stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale numeric columns (fit on train only)
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Balance the training data only
X_train_sm, y_train_sm = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

### 4. Modeling
```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                             eval_metric="logloss", random_state=42),
}
for name, model in models.items():
    model.fit(X_train_sm, y_train_sm)
```

### 5. Evaluation
![Confusion matrices](images/05_confusion_matrices.png)
![ROC curves](images/06_roc_curves.png)
![Feature importance](images/07_feature_importance.png)

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.746 | 0.515 | 0.743 | 0.608 | 0.828 |
| Random Forest | 0.770 | 0.559 | 0.650 | 0.601 | 0.817 |
| XGBoost | 0.756 | 0.530 | 0.714 | 0.608 | 0.823 |

**Recall** was prioritized over accuracy, since missing a customer who is about to churn is more costly to the business than a false alarm.

**Recommended model: Logistic Regression** - it had the best ROC-AUC (0.828) and recall (0.743) of the three, and is simpler to explain to a business team than XGBoost.

## Business Recommendations
- Offer incentives (discounts, added perks) to move month-to-month customers onto longer contracts.
- Bundle online security and tech support free for the first few months to reduce early churn.
- Give extra retention attention to customers in their first year, since this is the highest-risk period.

## Project Structure
```
customer-churn-prediction/
├── Customer_Churn_Prediction.ipynb   # full notebook: EDA + preprocessing + models + evaluation
├── images/                           # chart screenshots used in this README
├── model_comparison.csv              # final metrics table
└── README.md
```

## How to Run
1. Clone this repository.
2. Install dependencies: `pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn`
3. Open `Customer_Churn_Prediction.ipynb` in Jupyter Notebook or Google Colab and run all cells in order.
4. The dataset is loaded directly from a public URL in the notebook, so no manual download is required.

## Author
Manku Deepika
