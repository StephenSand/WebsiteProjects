#data from https://www.kaggle.com/datasets/leilaaliha/ecommerce-customers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.model_selection import GridSearchCV

# Reading data
df = pd.read_table('Ecommerce Customers',delimiter=',')

## Looking at the at the features
#pd.options.display.max_columns = None
#df.describe()
#df.iloc[0]

# Dropping unusable cols
df.drop(['Address','Email','Avatar'],axis=1,inplace=True)
df.dropna(inplace=True)


## Pairplot
sns.set_theme(palette="inferno")
pear = sns.pairplot(df)
#plt.show()
pear.savefig('reg_pairplot.png')

# For this dataset there is no need for scaling, normalization, or polynomial transformation

# Splitting data
X = df.drop('Yearly Amount Spent',axis=1)
y = df['Yearly Amount Spent']
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)


# LINEAR REGRESSION
linreg = LinearRegression().fit(X_train, y_train)
print(linreg.score(X_train, y_train))
# 0.9839128750292896
print(linreg.score(X_test, y_test))
# 0.9851050788797161
# Scoring
y_pred = linreg.predict(X_test)
rmse_score= rmse(y_test,y_pred)
print('Root mean squared error is {rmse_score:.2f}, about {error_size:.2f} percent of the mean value of y: {mean:.2f}'.format(rmse_score=rmse_score, error_size=(rmse_score/y.mean())*100, mean=y.mean() ) )
# Root mean squared error is 9.82, about 1.97 percent of the mean value of y: 499.31
# Visualizing the model predictions vs actual values
y_test1 = y_test.reset_index(drop=True)
scatter_fig = plt.figure()
sns.scatterplot(y_test1, label='Actual')
sns.scatterplot(y_pred, marker='x', label='Predicted')
scatter_fig.savefig('reg_actual_predicted.png')

# Plotting Prediction Error
from sklearn.metrics import PredictionErrorDisplay
display = PredictionErrorDisplay(y_true=y_test, y_pred=y_pred)
display.plot(scatter_kwargs={'color':'indigo'})
plt.title('Prediction Error')
plt.savefig('lr_predictionerror.png')
#plt.show()

## Other models - commented out

# RIDGE REGRESSION
#ridge = Ridge().fit(X_train, y_train)
#print(ridge.score(X_train, y_train))
# 0.9839061102851367
#print(ridge.score(X_test, y_test))
# 0.9851548733402657
# Parameter Tuning
#values = {'alpha': [.001,.01,.1,1,10]}
#grid_search = GridSearchCV(ridge, param_grid=values,scoring='neg_root_mean_squared_error')
#grid_search.fit(X_train, y_train)
#print("Best value of parameter 'alpha':", grid_search.best_params_)
# Best value of parameter 'alpha': {'alpha': 0.1}
#print("Best score:", grid_search.best_score_)
# Best score: -10.037072771806738
# Final model
#ridge = Ridge(alpha=grid_search.best_params_['alpha']).fit(X_train, y_train)
#y_pred = ridge.predict(X_test)
#rmse_score= rmse(y_test,y_pred)
#print('Root mean squared error is {rmse_score:.2f}, about {error_size:.2f} percent of the mean value of y: {mean:.2f}'.format(rmse_score=rmse_score, error_size=(rmse_score/y.mean())*100, mean=y.mean() ) )
# Root mean squared error is 9.82, about 1.97 percent of the mean value of y: 499.31


# LASSO REGRESSION
#lasso = Lasso().fit(X_train, y_train)
#print(lasso.score(X_train, y_train))
# 0.9834038371103452
#print(lasso.score(X_test, y_test))
# 0.9849091227579093
# Parameter Tuning
#values = {'alpha': [.001,.01,.1,1,10]}
#grid_search = GridSearchCV(lasso, param_grid=values, scoring='neg_root_mean_squared_error')
#grid_search.fit(X_train, y_train)
#print("Best value of parameter 'alpha':", grid_search.best_params_)
# Best value of parameter 'alpha': {'alpha': 0.1}
#print("Best score:", grid_search.best_score_)
# Best score: -10.03490283561515
# Final model
#lasso = Lasso(alpha=grid_search.best_params_['alpha']).fit(X_train, y_train)
#print(lasso.score(X_train, y_train))
# 0.9839061866290323
#print(lasso.score(X_test, y_test))
# 0.9851175494857103
#y_pred = lasso.predict(X_test)
#rmse_score= rmse(y_test,y_pred)
#print('Root mean squared error is {rmse_score:.2f}, about {error_size:.2f} percent of the mean value of y: {mean:.2f}'.format(rmse_score=rmse_score, error_size=(rmse_score/y.mean())*100, mean=y.mean() ) )
# Root mean squared error is 9.82, about 1.97 percent of the mean value of y: 499.31


