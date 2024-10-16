# Banking Churn Prediction Project  
This is my first 'real-world' project, using data from over 300,000 bank customer accounts:  
Based on data from the preceding six months, predict account termination (churn) in the next three months. The original project was for an Italian bank; I've redone it from the project description and original dataset (which is not available for obvious security reasons).

This project deals with two major issues: one technical, the other business-related.
- The technical issue is the extreme target imbalance: out of 301895 accounts, only 1586 (0.5%) terminated.
- The business issue relates to the ROI of amelioration: optimizing the threshold (of the probability-score).
The predictions identify (potentially) terminating customers for marketing group; ROI calculations show the best balance
between contact cost versus 'cost' of losing an account.

The scikit-learn DecisionTreeClassifier (DTC) model is used for most of the predicting, but other models are
examined: BalancedBaggingClassifier, RandomForestClassifier, XGboost, catBoost, lightGBM. Various resampling
techniques are tested also: undersampling, SMOTE. A comparison spreadsheet is in the'DecisionTreeClassifier' folder:
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
