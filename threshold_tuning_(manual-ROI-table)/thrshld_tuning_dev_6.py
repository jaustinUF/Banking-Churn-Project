# get metrics by bin
import pandas as pd
import joblib
from columnar import columnar  # used once (line 53)
from sklearn.metrics import get_scorer_names
import csv

# load pretrained model; model created/trained in thrshld_tuning_dev_2.py
model_data = joblib.load('DTC_model_data.pkl') # dictionary
model = model_data['model']                 # object instance of class DecisionTreeClassifier
X_test = model_data['X_test']               # Type: dataframe
y_test = model_data['y_test']               # Type: series
## --> now can work with trained model

# Predict probabilities: array of probabilities observation is in the positive class
y_probs = model.predict_proba(X_test)[:, 1] # numpy array
# Create a dataframe of true values and probabilities
results = pd.DataFrame({'actual': y_test, 'proba': y_probs})

## predicted value calculations
threshold = 0.5                             # threshold to put observation in predicted positive class
# add 'predicted' column to 'results' database based on 'threshold'
df = results.assign(predicted = lambda x: (x['proba'] > threshold).astype(int))
df = df.sort_values(by='proba', ascending=True) # Sort by predicted probabilities (not necessary for metrics)

## create bins and name
df['bins'] = pd.cut(df['proba'], bins=10)   # bin on 'proba' (adds 'bins' column)
df['bins_str'] = df['bins'].astype(str)     # add column with bin name in string format
# edit bin string name to string format '0.x-0.y'
df['bins_str'] = df['bins_str'].str.replace('[\(\)\[\]]', '', regex=True)
df['bins_str'] = df['bins_str'].str.replace(', ', '-', regex=True)
df['bins_str'] = df['bins_str'].str.replace('-0.001-0.1', '0.0-0.1', regex=True)

# get metrics
contact_cost = 1
acnt_rev = 40
# bins = ['0.0-0.1', '0.1-0.2', '0.2-0.3', '0.3-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
bins = ['0.9-1.0', '0.8-0.9', '0.7-0.8', '0.6-0.7', '0.5-0.6', '0.4-0.5', '0.3-0.4', '0.2-0.3', '0.1-0.2', '0.0-0.1']
headers = ['bin', 'size', 'AP', 'TP', 'TN', 'FP', 'FN', 'precision', 'recall', 'ROI']
data = []

# for b in reversed(bins):
for b in bins:
    bin_size = len(df[(df['bins_str'] == b)])
    AP = len(df[(df['actual'] == 1) & (df['bins_str'] == b)]) # actual positives
    TP = len(df[(df['actual'] == 1) & (df['predicted'] == 1) & (df['bins_str'] == b)])
    TN = len(df[(df['actual'] == 0) & (df['predicted'] == 0) & (df['bins_str'] == b)])
    FP = len(df[(df['actual'] == 0) & (df['predicted'] == 1) & (df['bins_str'] == b)])
    FN = len(df[(df['actual'] == 1) & (df['predicted'] == 0) & (df['bins_str'] == b)])
    precisn = round((0 if TP == 0 else TP/(TP + FP)) * 100, 1)
    recall = round((0 if TP == 0 else TP/(TP + FN)) * 100, 1)
    ROI = (FN * acnt_rev) + (FP * contact_cost)
    data.append([b, bin_size, AP, TP, TN, FP, FN, precisn, recall, ROI])
print(f'Metrics for threshold = {threshold}')
print(columnar(data, headers, no_borders=True ))
print()
# print(f'get_scorer_names\n{get_scorer_names()}') # https://scikit-learn.org/stable/modules/classification_threshold.html

# with open(f'Tables/DTC_thrsh_{threshold}.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerows([headers] + data)
#     print('File saved!')
