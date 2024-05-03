#data from https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier



#### Cleaning dataset

# Reading the files
train = pd.read_csv('train.csv', index_col=0)
test = pd.read_csv('test.csv', index_col=0)
complete = pd.concat([train,test],axis=0)
# Seeing whether to drop NaN values or fill with zeros
len(complete)
#129880
len(complete.dropna())
#129487
complete.dropna(inplace=True)
# Only about 400 rows are lost so it would be better to just drop them

### vis 0: iloc[0] ###
# To take a closer look at the features, the first row will be visualized
#pd.DataFrame(complete.iloc[0]).to_html().replace('\n','')
# It seems there are some text features that will have to be transformed to be used by the classifier. We can also drop 'id' as it's not useful.
complete.drop('id', axis=1, inplace=True)

# Closer look at the text features
for x in complete.columns:
    if len(complete[x].unique()) < 10:
        print(x,complete[x].unique())
    else: print(x,'more than 10 unique values')


# It looks like we can make True/False columns for the features with 2 unique values and make dummy matrices for the text features with more than 2 unique values

# Transforming text features to numerical features
for x in complete.columns:
    uniq_vals = complete[x].unique()
    if type(uniq_vals[0]) == type('potato'):
        if len(uniq_vals) == 2:
            complete[x].replace(uniq_vals,[0,1], inplace=True)
        else:
            dummies = pd.get_dummies(complete[x])
            complete = pd.concat([complete,dummies], axis=1)
            complete.drop(x, axis=1, inplace=True)




#### Splitting training and testing data
# Creating the X and Y dataframes
X = complete.drop('satisfaction',axis=1)
y = complete['satisfaction']
# Creating the training and testing dataframes
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

### vis 1: describe ##
# Time to take a better look at the featues now that they can be analyzed
#pd.options.display.max_columns = None
X.describe().to_html().replace('\n','')
### vis 1: pairplot ###
pear = sns.pairplot(X)
pear.savefig('class_pairplot.png')

# For this dataset scaling is not necessary and when we use scaling, the model is overfit
"""
# Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = scaler.transform(X_test)
"""



# Feature Selection

# SelectKBest with ANOVA f_classif
best = SelectKBest(score_func=f_classif, k=16)
X_train_best = best.fit_transform(X_train,y_train)
X_test_best = best.transform(X_test)
# To see the best features...
features = [int(x[1:]) for x in best.get_feature_names_out()]
feature_names = [X.columns[x] for x in features]
print(feature_names)



#### ML Models

# Function for making an ideal ml model
def make_model(model,params):
    clf = model
    clf.fit(X_train_best,y_train)
    grid_search = GridSearchCV(clf, param_grid=params, scoring='roc_auc')
    grid_search.fit(X_train_best, y_train)
    print(grid_search.best_params_)
    print('Score: ',grid_search.best_score_)
    clf.set_params(**grid_search.best_params_)
    print('Train score: ', clf.score(X_train_best, y_train))
    print('Test score: ', clf.score(X_test_best, y_test))
    return clf



# Logistic Regression
lr_params = {'C':[.001,.01,.1,1,10,100]}
lr = make_model(LogisticRegression(),lr_params)

# KNeighbors
knn_params = {'n_neighbors':[1,3,5,7,9]}
knn = make_model(KNeighborsClassifier(),knn_params)
# For KNeighbors the training score was better than the testing score, indicating overfitting

# Linear SVC
lsvc_params = {'C':[.001,.01,.1,1,10,100]}
lsvc = make_model(LinearSVC(), lsvc_params)

# RandomForest
rf_params = {'n_estimators': [25, 50, 100, 150], 'max_features': ['sqrt', 'log2', None], 'max_depth': [3, 6, 9], 'max_leaf_nodes': [3, 6, 9],}
rf = make_model(RandomForestClassifier(), rf_params)
# For RandomForest the training score was better than the testing score, indicating overfitting

# MLP
mlp_params = {'alpha': [0.0001, 0.05, .1]}
mlp = make_model(MLPClassifier(), mlp_params)

# Cross Validation Score

print(cross_val_score(mlp, X, y, cv=5))
print(cross_val_score(mlp, X, y, cv=5, scoring = 'roc_auc'))

# Classification Report
y_pred = mlp.predict(X_test_best)
cr = classification_report(y_test,y_pred)

### vis 3: classification report ###
print(cr)


# Making this classification report html friendly
#...
"""
table border="1" class="dataframe">  <thead>    <tr style="text-align: right;">      <th></th>      <th>precision</th>      <th>recall</th>      <th>f1-score</th>      <th>support</th>    </tr>    <tr>      <th>measures</th>      <th></th>      <th></th>      <th></th>      <th></th>    </tr>  </thead>  <tbody>    <tr>      <th>0</th>      <td>0.89</td>      <td>0.97</td>      <td>0.93</td>      <td>18310</td>    </tr>    <tr>      <th>1</th>      <td>0.96</td>      <td>0.84</td>      <td>0.90</td>      <td>14062</td>    </tr>    <tr>      <th>accuracy</th>      <td>None</td>      <td>None</td>      <td>0.92</td>      <td>32372</td>    </tr>    <tr>      <th>macro_avg</th>      <td>0.92</td>      <td>0.91</td>      <td>0.91</td>      <td>32372</td>    </tr>    <tr>      <th>weighted_avg</th>      <td>0.92</td>      <td>0.92</td>      <td>0.91</td>      <td>32372</td>    </tr>  </tbody></table>
"""
