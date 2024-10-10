# Banking Churn Prediction Project
This is my first 'real-world' project: based on data from the preceding six months of over 300,000 bank accounts,
predict account termination in the next three months. The project was originally done 
for an Italian bank by my ML mentor; I've redone it from the project description and original dataset.
(dataset not available as it contains real personal information).

This project deals with two major issues: one technical, the other business-related.
- The technical issue is the extreme target imbalance: out of 301895 accounts, only 1586 (0.5%) terminated.
- The business issue relates to the ROI of amelioration: optimizing the threshold (of the probability-score).
The predictions show marketing (potentially) terminating customers; ROI calculations show the best balance
between cost of contact and 'cost' of losing an account.

The scikit-learn DecisionTreeClassifier (DTC) model is used for most of the predicting, but other models are
examined: BalancedBaggingClassifier, RandomForestClassifier, XGboost, catBoost, lightGBM. Various resampling
techniques are tested also: undersampling, SMOTE. Comparisons:
'DecisionTreeClassifier' folder
    - DecisionTreeClassifier metrics.docx

Data preparation varies by model: one-hot encoding, missing values, type change, etc.

DTC model tuning (cumulative gains charts): 'model_tuning' folder
        - comparison of max_depth argument changes
        
ROI table from manual calculations: 'threshold_tuning_(manual-ROI-table)' folder
    - DecisionTreeClassifier model
        - 'thrshld_tuning_dev_6.py'
        - 'thrshld_tuning_dev_6_grouped.py'

Cost-sensitive learning - TunedThresholdClassifierCV model
    - find 'best' threshold for cost-related  outcomes like ROIs        
    - 'Tune In-Decision Threshold Optimization' folder
        - dtc_baseline_cost_sen_learning.py'
        - cost-sensitive_learning.py
