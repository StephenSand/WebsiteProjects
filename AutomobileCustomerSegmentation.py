# data from https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.feature_selection import f_classif, SelectKBest

## Read datasets
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# Concat train & test because need to clean nans
df = pd.concat([train,test],axis=0)
df.reset_index(drop=True,inplace=True)
df.dropna(inplace=True)

### vis 0: describe ###
df.describe().to_html().replace('\n','')

### vis 1: iloc[0] ###
pd.DataFrame(df.iloc[0]).to_html().replace('\n','')


### vis 2: pairplot ###
pear = sns.pairplot(df)
pear.savefig('mclass_pairplot_before.png')

## Cleaning
for x in df.columns:
    print(x)
    if len(df[x].unique()) < 10:
        print(df[x].unique())
    else: print('more than 10 unique values')


df.drop(['ID'], axis=1, inplace=True)
df['Spending_Score'] = df['Spending_Score'].replace({'Low':1,'Average':2,'High':3})
df['Var_1'] = df['Var_1'].replace({'Cat_4':4, 'Cat_6':6, 'Cat_7':7, 'Cat_3':3, 'Cat_1':1, 'Cat_2':2, 'Cat_5':5})
df['Ever_Married'] = df['Ever_Married'].replace({'No':False,'Yes':True})
df['Graduated'] = df['Graduated'].replace({'No':False,'Yes':True})
df['Gender'] = df['Gender'].replace({'Male':False,'Female':True})
profession = pd.get_dummies(df['Profession'])
df = pd.concat([df,profession], axis=1)
df.drop('Profession',axis=1,inplace=True)

### vis 3: pairplot ###
pear = sns.pairplot(df)
pear.savefig('mclass_pairplot_after.png')

## Splitting Data
X = df.drop('Segmentation',axis=1)
y = df['Segmentation']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

## Scaling
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Feature Selection with SelectKBest
best = SelectKBest(score_func=f_classif, k=5)
best.fit(X_train_scaled,y_train)
X_train_best = best.transform(X_train_scaled)
X_test_best = best.transform(X_test_scaled)
print(best.get_feature_names_out())
best_cols = [int(x[1:]) for x in best.get_feature_names_out()]
best_features = [X.columns[x] for x in best_cols]
print(best_features)

#### Training the models
svc = SVC(kernel='poly',probability=True,random_state=0,decision_function_shape='ovo')
svc.fit(X_train_best,y_train)
pred = svc.predict(X_test_best)
score = f1_score(y_test,pred,average='macro')
print(score)


def tune_svc(x=1.0, y=3, z='scale'):
    svc = SVC(kernel='poly',probability=True,random_state=0,decision_function_shape='ovo', C=x, degree=y, gamma=z)
    svc.fit(X_train_best,y_train)
    pred = svc.predict(X_test_best)
    score = f1_score(y_test,pred,average='macro')
    print(score)
    return svc


"""
#svc takes too long to tune...
#these are the ideal parameters
values = {'C':[.05,.1,.5], 'degree':[5,7,9], 'gamma':[1,5,10]}
grid_search = GridSearchCV(svc, param_grid=values, scoring='f1_macro')
grid_search.fit(X_train_best, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best F1 score:", grid_search.best_score_)
"""

# Random forest
rfc = RandomForestClassifier()
rfc.fit(X_train_best,y_train)
rfc_pred = rfc.predict(X_test_best)
score = f1_score(y_test,rfc_pred,average='macro')
print(score)

# Neural Networks MLP
mlp = MLPClassifier(activation= 'relu', alpha= 0.005, hidden_layer_sizes= (50, 50), learning_rate= 'adaptive', solver= 'adam')
mlp.fit(X_train_best,y_train)
mlp_pred = mlp.predict(X_test_best)
score = f1_score(y_test,mlp_pred,average='macro')
print(score)

"""
#these are the ideal values for the MLP
values = {
    'hidden_layer_sizes': [(50,50,), (100,), (100,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.005],
    'learning_rate': ['constant','adaptive'],
}
grid_search = GridSearchCV(mlp, param_grid=values, scoring='f1_macro')
grid_search.fit(X_train_best, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best F1 score:", grid_search.best_score_)
"""

# Bagging Classifier
bc = BaggingClassifier(estimator=mlp)
bc.fit(X_train_best,y_train)
bc_pred = bc.predict(X_test_best)
score = f1_score(y_test,bc_pred,average='macro')
print(score)


################# notes #############
#so far the mlp classifier is the best...
#the scaler and the selectkbest help...

y_test.reset_index(drop=True, inplace=True)
predictions = pd.Series(bc_pred)
accuracy = y_test == predictions
accuracy.describe()
# Result: 1016/2205 predictions were accurate giving the bagging classifier an accuracy of 0.46077097505668935 or roughly 46%.
# This is not an ideal score. I believe this model could improve with the addition of more data.
