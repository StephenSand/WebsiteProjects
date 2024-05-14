# data from https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.feature_selection import f_classif, SelectKBest
from xgboost import XGBClassifier

# Function for getting the train & test scores
def get_scores(clf):
    train_prd = clf.predict(X_train_best)
    train_score = f1_score(y_train,train_prd,average='macro')
    test_prd = clf.predict(X_test_best)
    test_score = f1_score(y_test,test_prd,average='macro')
    print('Train: ',train_score)
    print('Test: ',test_score)
    return test_prd

# Function for finding the best parameters for ml models
def get_params(clf, params):
    grid_search = GridSearchCV(clf, param_grid=params, scoring='f1_macro', cv=3, n_jobs=-1)
    grid_search.fit(X_train_best, y_train)
    print(grid_search.best_params_)
    print(grid_search.best_score_)
    return grid_search.best_params_


## Read datasets
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

# Concat train & test because need to clean nans
df = pd.concat([train,test],axis=0)
df.reset_index(drop=True,inplace=True)
df.dropna(inplace=True)

## Taking a look at the features
#df.describe()
#df.iloc[0]

## Cleaning
alpha_feats = df.select_dtypes(exclude=np.number)
for x in alpha_feats.columns:
    print(x)
    if len(df[x].unique()) < 10:
        print(df[x].unique())
    else: print('more than 10 unique values')

#Gender
#['Male' 'Female']
#Ever_Married
#['No' 'Yes']
#Graduated
#['No' 'Yes']
#Profession
#['Healthcare' 'Engineer' 'Lawyer' 'Artist' 'Doctor' 'Homemaker'
# 'Entertainment' 'Marketing' 'Executive']
#Spending_Score
#['Low' 'High' 'Average']
#Var_1
#['Cat_4' 'Cat_6' 'Cat_7' 'Cat_3' 'Cat_1' 'Cat_2' 'Cat_5']
#Segmentation
#['D' 'B' 'C' 'A']

# Silent downcasting will be deprecated in a future release of Pandas
pd.set_option('future.no_silent_downcasting', False)
# Changing alphabetic features to numeric features
df.drop(['ID'], axis=1, inplace=True)
df['Spending_Score'] = df['Spending_Score'].replace({'Low':1,'Average':2,'High':3})
df['Var_1'] = df['Var_1'].replace({'Cat_4':4, 'Cat_6':6, 'Cat_7':7, 'Cat_3':3, 'Cat_1':1, 'Cat_2':2, 'Cat_5':5})
df['Ever_Married'] = df['Ever_Married'].replace({'No':False,'Yes':True})
df['Graduated'] = df['Graduated'].replace({'No':False,'Yes':True})
df['Gender'] = df['Gender'].replace({'Male':False,'Female':True})
profession = pd.get_dummies(df['Profession'])
df = pd.concat([df,profession], axis=1)
df.drop('Profession',axis=1,inplace=True)

## Pairplot
sns.set_theme(palette="inferno")
#this takes a long time
pear = sns.pairplot(df)
pear.savefig('mclass_pairplot.png')

## Splitting Data
X = df.drop('Segmentation',axis=1)
y = df['Segmentation']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

## Scaling
#scaler = MinMaxScaler().fit(X_train)
#scaler = StandardScaler().fit(X_train)
scaler = RobustScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

## Feature Selection with SelectKBest
best = SelectKBest(score_func=f_classif, k=5)
best.fit(X_train_scaled,y_train)
X_train_best = best.transform(X_train_scaled)
X_test_best = best.transform(X_test_scaled)
best_cols = [int(x[1:]) for x in best.get_feature_names_out()]
best_features = [X.columns[x] for x in best_cols]
print(best_features)
#['Ever_Married', 'Age', 'Spending_Score', 'Artist', 'Healthcare']

# Changing the string target values to numbers for xgb
y_train = y_train.replace({'A':0,'B':1,'C':2,'D':3})
y_test = y_test.replace({'A':0,'B':1,'C':2,'D':3})

## Training the models
# SVC
svc = SVC(random_state=0)
svc.fit(X_train_best,y_train)
# Gridsearch Parameter Tuning
# SVC is kind of a pain to tune... will have to determine kernel first then the rest
svc_grid_search = get_params(svc, {'kernel':['rbf','linear','poly']})
#{'kernel': 'rbf'}
#0.4524008358780387
# Now that we have the kernel we can continue with the other parameters
svc = SVC(random_state=0, kernel='rbf')
svc.fit(X_train_best,y_train)
svc_params = {'C':[0.1,1,10], 'gamma':[1,10,30]}
svc_best_params = get_params(svc, svc_params)
#{'C': 1, 'gamma': 10}
#0.46075208227240694
svc_best_params['random_state'] = 0
svc_best_params['kernel'] = 'rbf'
svc = SVC(**svc_best_params)
svc.fit(X_train_best,y_train)
svc_prd = get_scores(svc)
#Train:  0.4793373948888611
#Test:  0.4591707013979063



# Random forest
rfc = RandomForestClassifier()
rfc.fit(X_train_best,y_train)
rfc_params = {
    'n_estimators': [25, 50, 100, 150], 
    'max_features': ['sqrt', 'log2', None], 
    'max_depth': [3, 6, 9], 
    'max_leaf_nodes': [3, 6, 9], 
}
rfc_best_params = get_params(rfc, rfc_params)
#{'max_depth': 6, 'max_features': None, 'max_leaf_nodes': 9, 'n_estimators': 100}
#0.45774985333405543
rfc = RandomForestClassifier(**rfc_best_params)
rfc.fit(X_train_best,y_train)
rfc_prd = get_scores(rfc)
#Train:  0.4785066413425025
#Test:  0.4497253548060093



# XG Boost
xgb = XGBClassifier()
xgb.fit(X_train_best,y_train)
# parameter tuning
xgb_params = {
    'max_depth': [3,6,9],
    'learning_rate': [0.3,0.1,0.05],
    'subsample': [0.5,0.75,1],
    'n_estimators':[100,150,200],
    'colsample_bytree':[0.5, 0.75, 1],
    'min_child_weight':[1,5,15]
}
best_xgb_params = get_params(xgb, xgb_params)
#{'colsample_bytree': 0.75, 'learning_rate': 0.1, 'max_depth': 3, 'min_child_weight': 5, 'n_estimators': 150, 'subsample': 0.5}
#0.45774985333405543
# final model
xgb = XGBClassifier(**best_xgb_params)
xgb.fit(X_train_best,y_train)
# training & testing scores
xgb_prd = get_scores(xgb)
#Train:  0.4785066413425025
#Test:  0.4497253548060093



# Neural Networks MLP
mlp = MLPClassifier()
mlp.fit(X_train_best,y_train)
mlp_params = {
    'hidden_layer_sizes': [(50,50,), (100,), (100,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.005],
    'learning_rate': ['constant','adaptive'],
}
mlp_best_params = get_params(mlp,mlp_params)
#{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50), 'learning_rate': 'constant', 'solver': 'adam'}
#0.46181761034180346
mlp = MLPClassifier(**mlp_best_params)
mlp.fit(X_train_best,y_train)
mlp_prd = get_scores(mlp)
#Train:  0.4753354595212361
#Test:  0.4571859077238673


# Bagging Classifier
bc = BaggingClassifier(estimator=svc)
bc.fit(X_train_best,y_train)
bc_prd = get_scores(bc)
#Train:  0.47810352174190934
#Test:  0.46399446166940495



## Visualizing pr & roc auc curves
# Changing y values back to original names and types for plotting
y_train = y_train.replace({0:'A',1:'B',2:'C',3:'D'})
y_test = y_test.replace({0:'A',1:'B',2:'C',3:'D'})
y_dums = pd.get_dummies(y_test)
y_dums.reset_index(drop=True, inplace=True)
prd_dums = pd.get_dummies(bc_prd)
prd_dums.columns = ['A1','B1','C1','D1']

# PR #
probs = pd.DataFrame(bc.predict_proba(X_test_best))
precision_A, recall_A, pr_threshold_A = precision_recall_curve(y_dums['A'], probs[0])
precision_B, recall_B, pr_threshold_B = precision_recall_curve(y_dums['B'], probs[1])
precision_C, recall_C, pr_threshold_C = precision_recall_curve(y_dums['C'], probs[2])
precision_D, recall_D, pr_threshold_D = precision_recall_curve(y_dums['D'], probs[3])


pr = plt.figure()
sns.lineplot(x=recall_A, y=precision_A, **{'label':'A'})
sns.lineplot(x=recall_B, y=precision_B, **{'label':'B'})
sns.lineplot(x=recall_C, y=precision_C, **{'label':'C'})
sns.lineplot(x=recall_D, y=precision_D, **{'label':'D'})
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.title('Precision vs. Recall Curve')
pr.savefig('mclass_pr.png')



# ROC AUC #
false_pos_A, true_pos_A, threshold_A = roc_curve(y_dums['A'], prd_dums['A1'])
false_pos_B, true_pos_B, threshold_B = roc_curve(y_dums['B'], prd_dums['B1'])
false_pos_C, true_pos_C, threshold_C = roc_curve(y_dums['C'], prd_dums['C1'])
false_pos_D, true_pos_D, threshold_D = roc_curve(y_dums['D'], prd_dums['D1'])

roc_auc_A = auc(false_pos_A, true_pos_A)
roc_auc_B = auc(false_pos_B, true_pos_B)
roc_auc_C = auc(false_pos_C, true_pos_C)
roc_auc_D = auc(false_pos_D, true_pos_D)

roc_vis = plt.figure()

sns.lineplot(x=[0, 1], y=[0, 1], linestyle='--', label='No Skill')
sns.lineplot(x=false_pos_A, y=true_pos_A, label='A: '+str(roc_auc_A))
sns.lineplot(x=false_pos_B, y=true_pos_B, label='B: '+str(roc_auc_B))
sns.lineplot(x=false_pos_C, y=true_pos_C, label='C: '+str(roc_auc_C))
sns.lineplot(x=false_pos_D, y=true_pos_D, label='D: '+str(roc_auc_D))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(title='AUC')
plt.title('ROC AUC Curve')
roc_vis.savefig('mclass_roc_auc.png')











