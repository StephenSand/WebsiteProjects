#data from https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

# Function for making an ideal ml model
def make_model(model,params):
    clf = model
    clf.fit(X_train_best,y_train)
    grid_search = GridSearchCV(clf, param_grid=params, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train_best, y_train)
    print(grid_search.best_params_)
    print('Score: ',grid_search.best_score_)
    clf.set_params(**grid_search.best_params_)
    print('Train score: ', clf.score(X_train_best, y_train))
    print('Test score: ', clf.score(X_test_best, y_test))
    return clf


## Cleaning dataset

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

## Taking a look at the features
#complete.iloc[0]
# It seems there are some text features that will have to be transformed to be used by the classifier. We can also drop 'id' as it's not useful.
complete.drop('id', axis=1, inplace=True)

# Silent downcasting is being deprecated in the future
pd.set_option('future.no_silent_downcasting', False)

# Closer look at the text features
alpha_features = complete.select_dtypes(exclude=np.number)
for x in alpha_features.columns:
    if len(alpha_features[x].unique()) < 10:
        print(x,alpha_features[x].unique())
    else: print(x,'more than 10 unique values')


# It looks like we can make True/False columns for the features with 2 unique values and make dummy matrices for the text features with more than 2 unique values

# Transforming text features to numerical features
for x in alpha_features.columns:
    uniq_vals = alpha_features[x].unique()
    if len(uniq_vals) == 2:
        complete[x] = complete[x].replace(uniq_vals,[0,1])
    else:
        dummies = pd.get_dummies(complete[x])
        complete = pd.concat([complete,dummies], axis=1)
        complete.drop(x, axis=1, inplace=True)


## Pairplot
sns.set_theme(palette="inferno")
pear = sns.pairplot(complete)
pear.savefig('class_pairplot.png')

# For this dataset scaling is not necessary and when we use scaling, the model is overfit

## Splitting training and testing data
# Creating the X and Y dataframes
X = complete.drop('satisfaction',axis=1)
y = complete['satisfaction']
# Creating the training and testing dataframes
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

# Time to take a better look at the featues now that they can be analyzed #
#pd.options.display.max_columns = None
X.describe()

## Feature Selection
# SelectKBest with ANOVA f_classif
best = SelectKBest(score_func=f_classif, k=16)
X_train_best = best.fit_transform(X_train,y_train)
X_test_best = best.transform(X_test)
# Seeing the best features
features = [x for x in best.get_feature_names_out()]
print(features)
#['Customer Type', 'Type of Travel', 'Flight Distance', 'Inflight wifi service', 'Food and drink', 'Online boarding', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness', 'Business', 'Eco']


## ML Models
# Logistic Regression
lr_params = {'C':[.001,.01,.1,1,10,100]}
lr = make_model(LogisticRegression(),lr_params)
#{'C': 0.001}
#Score:  0.8842808802315341
#Train score:  0.7992380167842249
#Test score:  0.8000123563573458
# Plotting Decision Boundaries #

# KNeighbors
knn_params = {'n_neighbors':[1,3,5,7,9]}
knn = make_model(KNeighborsClassifier(),knn_params)
#{'n_neighbors': 9}
#Score:  0.9140536688971711
#Train score:  0.8803583380528239
#Test score:  0.8559866551340665
#Train Score > Test Score == Overfit

# Linear SVC
lsvc_params = {'C':[.001,.01,.1,1,10,100]}
lsvc = make_model(LinearSVC(), lsvc_params)
#{'C': 0.1}
#Score:  0.886876287265153
#Train score:  0.5714977089018174
#Test score:  0.5719448906462374

# RandomForest
rf_params = {'n_estimators': [25, 50, 100, 150], 'max_features': ['sqrt', 'log2', None], 'max_depth': [3, 6, 9], 'max_leaf_nodes': [3, 6, 9],}
rf = make_model(RandomForestClassifier(), rf_params)
#{'max_depth': 6, 'max_features': 'log2', 'max_leaf_nodes': 9, 'n_estimators': 150}
#Score:  0.9631764029545014
#Train score:  0.9999691087885496
#Test score:  0.9602434202397133
#Train Score > Test Score == Overfit

# MLP
mlp_params = {'alpha': [0.0001, 0.05, .1]}
mlp = make_model(MLPClassifier(), mlp_params)
#{'alpha': 0.0001}
#Score:  0.9757166601292045
#Train score:  0.9142665911548165
#Test score:  0.9151736068207093
mlp.fit(X_train_best,y_train)


# Cross Validation Score
print(cross_val_score(mlp, X, y, cv=5))
print(cross_val_score(mlp, X, y, cv=5, scoring = 'roc_auc'))

# Classification Report
y_pred = mlp.predict(X_test_best)
cr = classification_report(y_test,y_pred)
print(cr)

## Precision-Recall Curve
# Predicting y for X_test_best
probs = mlp.predict_proba(X_test_best)
y_preds = probs[:,1]
# Getting the precision and recall values
precision, recall, pr_threshold = precision_recall_curve(y_test, y_preds)
# Plotting with sns
pr_vis = plt.figure()
sns.lineplot(x=recall, y=precision).set_title('Precision vs. Recall')
plt.ylabel('Precision')
plt.xlabel('Recall')
pr_vis.savefig('binaryclass_pr_curve.png')


## ROC AUC Curve
# Getting the false positive and true positive rates
false_pos, true_pos, threshold = roc_curve(y_test, y_preds)
# Getting the roc_auc
roc_auc = auc(false_pos, true_pos)
# Plotting
roc_vis = plt.figure()
sns.lineplot(x=false_pos, y=true_pos, label=roc_auc)
sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--', label='No Skill').set_title('ROC AUC Curve')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
roc_vis.savefig('binaryclass_roc_curve.png')



