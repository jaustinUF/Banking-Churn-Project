import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from CGC_func import cum_gaines

max_depth = 14
plot_name = f'DTC CGC max_depth={max_depth} '
# save_plot = False
save_plot = True

# config wide display
pd.set_option('display.width', None)        # No wrapping to next line
pd.set_option('display.max_columns', None)  # Display all columns

data = pd.read_csv('../Churn_Banking_Modeling_ENG.csv') # get dataset
target = 'flag_request_closure'                         # define target
features_list = data.columns.drop(['customer_id', target]).tolist()   # list of feature names

# get non-numeric features
categorical_features = data.select_dtypes(include=['object']).columns.tolist()
categorical_features.remove(target)
# remove non-numeric features from features list
features_list = [f for f in features_list if f not in categorical_features]
for col in features_list:                               # (features now all numeric)
    data[col] = data[col].fillna(0)                     #    so fill in missing values with zero
# check features data set: only numeric features, with no missing values
# df_test = pd.DataFrame(data[features_list]); print(df_info(df_test, 15))
# prepare target encode to 0/1
le = LabelEncoder()
data[target] = le.fit_transform(data[target])
# Split data into standard X/y, train/test portions ... 80/20 split is common
X_train, X_test, y_train, y_test = train_test_split(data[features_list], data[target], test_size=0.2, random_state=42)
# create and train  model
print(f'y_train (dependent variable) value counts\n{y_train.value_counts()}\n')
model = DecisionTreeClassifier(random_state=42, max_depth=max_depth, min_samples_split=10, min_samples_leaf=5)
model.fit(X_train, y_train)

cum_gaines(model, X_test, y_test, plot_name, save_plot)













