#data from https://www.kaggle.com/datasets/leilaaliha/ecommerce-customers
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV

# Reading data
df = pd.read_table('Ecommerce Customers',delimiter=',')

### vis 0: describe ###
pd.options.display.max_columns = None
df.describe().to_html().replace('\n','')

### vis 1: iloc[0] ###
pd.DataFrame(df.iloc[0]).to_html().replace('\n','')

# Dropping unusable cols
df.drop(['Address','Email','Avatar'],axis=1,inplace=True)
df.dropna(inplace=True)

# Data Visualization
pear = sns.pairplot(df)
#plt.show()

### vis 2: pairplot ###
pear.savefig('reg_pairplot.png')

# For this dataset there is no need for scaling, normalization, or polynomial transformation

# Splitting data
X = df.drop('Yearly Amount Spent',axis=1)
y = df['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)
for x in X.columns:
    df.plot.scatter(x,'Yearly Amount Spent')
    plt.savefig(x+'.png')
    plt.show()
    plt.close()



# LINEAR REGRESSION
linreg = LinearRegression().fit(X_train, y_train)
print(linreg.score(X_train, y_train))
print(linreg.score(X_test, y_test))
# Scoring
y_pred = linreg.predict(X_test)
rmse_score= mse(y_test,y_pred,squared=False)
print('Root mean squared error is {rmse_score:.2f}, about {error_size:.2f} percent of the mean value of y: {mean:.2f}'.format(rmse_score=rmse_score, error_size=(rmse_score/y.mean())*100, mean=y.mean() ) )

# RIDGE REGRESSION
ridge = Ridge().fit(X_train, y_train)
print(ridge.score(X_train, y_train))
print(ridge.score(X_test, y_test))
# Parameter Tuning
values = {'alpha': [.001,.01,.1,1,10]}
grid_search = GridSearchCV(ridge, param_grid=values,scoring='neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Best value of parameter 'alpha':", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
# Final model
ridge = Ridge(alpha=grid_search.best_params_['alpha']).fit(X_train, y_train)
from sklearn.metrics import mean_squared_error as mse
y_pred = ridge.predict(X_test)
rmse_score= mse(y_test,y_pred,squared=False)
print('Root mean squared error is {rmse_score:.2f}, about {error_size:.2f} percent of the mean value of y: {mean:.2f}'.format(rmse_score=rmse_score, error_size=(rmse_score/y.mean())*100, mean=y.mean() ) )


# LASSO REGRESSION
lasso = Lasso().fit(X_train, y_train)
print(lasso.score(X_train, y_train))
print(lasso.score(X_test, y_test))
# Parameter Tuning
values = {'alpha': [.001,.01,.1,1,10]}
grid_search = GridSearchCV(lasso, param_grid=values, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)
print("Best value of parameter 'alpha':", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
# Final model
lasso = Lasso(alpha=grid_search.best_params_['alpha']).fit(X_train, y_train)
print(lasso.score(X_train, y_train))
print(lasso.score(X_test, y_test))
y_pred = lasso.predict(X_test)
rmse_score= mse(y_test,y_pred,squared=False)
print('Root mean squared error is {rmse_score:.2f}, about {error_size:.2f} percent of the mean value of y: {mean:.2f}'.format(rmse_score=rmse_score, error_size=(rmse_score/y.mean())*100, mean=y.mean() ) )


### Conclusion
"""
It seems the best classifier turned out to be the Lasso classifier.
With a train score of 0.983 and a test score of about 0.985, the model is not overfit.
With a root mean squared error of 9.82, the model is considered quite accurate as it may vary by around 2%.
