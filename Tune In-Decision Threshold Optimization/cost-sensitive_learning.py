# source information (original and mine) in README.md
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (make_scorer, confusion_matrix)
from sklearn.model_selection import TunedThresholdClassifierCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 26120

data = pd.read_excel("data/Telco_customer_churn.xlsx")
drop_cols = [
    "Count", "Country", "State", "Lat Long", "Latitude", "Longitude",
    "Zip Code", "Churn Value", "Churn Score", "CLTV", "Churn Reason"
]
data.drop(columns=drop_cols, inplace=True)
# print(data); print(f"\ninfo\n{data.info()}")
# Preprocess the data
data["Churn Label"] = data["Churn Label"].map({"Yes": 1, "No": 0})  # target to numeric
data.drop(columns=["Total Charges"], inplace=True)                  # ?? not dropped in original drop?
# split into X/y train/test datasets
X_train, X_test, y_train, y_test = train_test_split(
    data.drop(columns=["Churn Label"]),                             # get features (drop target)
    data["Churn Label"],                                            # target
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=data["Churn Label"],
)
# discussion of syntax below: https://chatgpt.com/c/5be7c805-63ed-4a9d-a612-17a46e9df0ef
# preprocess
preprocessor = ColumnTransformer(
    transformers=[("one_hot", OneHotEncoder(handle_unknown="infrequent_if_exist"),
                   selector(dtype_include="object"))],
    remainder="passthrough",
)
# pipeline to preprocess then instantiate Random Forest model
org_churn_model = make_pipeline(preprocessor, RandomForestClassifier(random_state=RANDOM_STATE))
# un-tuned model
org_churn_model.fit(X_train.drop(columns=["CustomerID"]), y_train)
# good discussion/process of assigning cost to mis-classifications in notebook at this point
#   See REAdME.md for location.
# quick check on what the one-hot encoding does: from 20 columns to 1173!
print(X_train.shape)
X_train_transformed = preprocessor.fit_transform(X_train.drop(columns=["CustomerID"]))
print(X_train_transformed.shape)

def cost_func(y, y_pred, neg_label, pos_label):
    cm = confusion_matrix(y, y_pred, labels=[neg_label, pos_label]) # see README.md for confusion matrix format
    cost_matrix = np.array([[0, -80], [0, 200]])
    return np.sum(cm * cost_matrix)
# note: 'make_scorer' expects custom func like 'cost_func'; knows how to supply arguments
cost_scorer = make_scorer(cost_func, neg_label=0, pos_label=1)

# tuned model
tuned_churn_model = TunedThresholdClassifierCV(
    org_churn_model,
    scoring=cost_scorer,
    store_cv_results=True,
)
tuned_churn_model.fit(X_train.drop(columns=["CustomerID"]), y_train)

# Calculate the profit on the test set
original_model_profit = cost_scorer(
    org_churn_model, X_test.drop(columns=["CustomerID"]), y_test
)
tuned_model_profit = cost_scorer(
    tuned_churn_model, X_test.drop(columns=["CustomerID"]), y_test
)
# print profit
print(f"Original model profit: {original_model_profit}")
print(f"Tuned model profit: {tuned_model_profit}")
# plot profit
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(
    tuned_churn_model.cv_results_["thresholds"],
    tuned_churn_model.cv_results_["scores"],
    marker="o",
    markersize=3,
    linewidth=1e-3,
    color="#c0c0c0",
    label="Objective score (using cost-matrix)",
)
ax.plot(
    tuned_churn_model.best_threshold_,
    tuned_churn_model.best_score_,
    "^",
    markersize=10,
    color="#ff6700",
    label="Optimal cut-off point for the business metric",
)
ax.legend()
ax.set_xlabel("Decision threshold (probability)")
ax.set_ylabel("Objective score (using cost-matrix)")
ax.set_title("Objective score as a function of the decision threshold")
plt.show()