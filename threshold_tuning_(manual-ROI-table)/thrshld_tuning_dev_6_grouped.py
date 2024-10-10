# get metrics by bin
import pandas as pd
import joblib
from columnar import columnar  # used once (line 53)
from sklearn.metrics import get_scorer_names
import csv

# config wide display
pd.set_option('display.width', None)        # No wrapping to next line
pd.set_option('display.max_columns', None)  # Display all columns

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
bins = ['0.9-1.0', '0.8-0.9', '0.7-0.8', '0.6-0.7', '0.5-0.6', '0.4-0.5', '0.3-0.4', '0.2-0.3', '0.1-0.2', '0.0-0.1']

grouped = df.groupby('bins_str').agg(
    bin_size=('bins_str', 'size'),
    AP=('actual', lambda x: sum(x == 1)),
    TP=('predicted', lambda x: sum((df['actual'] == 1) & (x == 1))),
    TN=('predicted', lambda x: sum((df['actual'] == 0) & (x == 0))),
    FP=('predicted', lambda x: sum((df['actual'] == 0) & (x == 1))),
    FN=('predicted', lambda x: sum((df['actual'] == 1) & (x == 0))),
)
# Calculate precision, recall
grouped['precision'] = round((grouped['TP'] / (grouped['TP'] + grouped['FP']).replace(0, 1)) * 100, 1)
grouped['recall'] = round((grouped['TP'] / (grouped['TP'] + grouped['FN']).replace(0, 1)) * 100, 1)

# Sort by 'bins_str' in descending order before calculating cumulative sums
grouped = grouped.reindex(bins).reset_index()

# Calculate cumulative sums and ROI
grouped['size_cum'] = grouped['bin_size'].cumsum()
grouped['FN_cum'] = grouped['FN'].cumsum()
grouped['FP_cum'] = grouped['FP'].cumsum()
grouped['ROI_FN'] = (grouped['FN_cum'] * acnt_rev) - (grouped['FP_cum'] * contact_cost)
grouped['ROI_size'] = (grouped['bin_size'] * acnt_rev) - (grouped['FP_cum'] * contact_cost)

print(f'grouped dataframe:\n{grouped.to_string(index=False)}')
ROI_table = grouped[['bins_str', 'bin_size', 'FP', 'FN', 'size_cum', 'FN_cum', 'FP_cum', 'ROI_FN', 'ROI_size']]
print(f'\n\nROI_table:\n{ROI_table.to_string(index=False)}')


# headers = ['bin', 'size', 'AP', 'TP', 'TN', 'FP', 'FN', 'precision', 'recall', 'ROI']

