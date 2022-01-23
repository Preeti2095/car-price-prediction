import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import  numpy as np

data = pd.read_csv('car price data/car data.csv')

# print(data.head())
# print(data.shape)
# print(data.isnull().sum())
# print((data.info()))
# for i in data.columns:
#     if data[i].dtype=='object':
#         print("uniquie " +i+" are: "+ str(data[i].nunique()))
#         print("uniquie " + i + " are: " + str(data[i].unique()))
#
# print(data.columns)
final_dataset = data[['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven',  'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
# print(final_dataset.head())
final_dataset['current_year']=2021
final_dataset['no_year']=final_dataset['current_year']-final_dataset['Year']
# print(final_dataset.head())
final_dataset.drop(['Year', 'current_year'], axis=1, inplace=True)
# print(final_dataset.head())
final_dataset = pd.get_dummies(final_dataset, drop_first=True)
# print(final_dataset.head())
# print(final_dataset.corr)
# sns.pairplot(final_dataset)


corrmat = final_dataset.corr()
top_corr_mat = corrmat.index
# plt.figure(figsize=(20,10))
g=sns.heatmap(final_dataset[top_corr_mat].corr(), annot=True, cmap='RdYlGn')
# plt.show()

#independent and dependent features
X=final_dataset.iloc[:, 1:]
y=final_dataset.iloc[:, 0]

#Linear Regression Model
from sklearn.linear_model import LinearRegression
regression=LinearRegression()

regression.fit(X_train,y_train)
pred=regression.predict(X_test)

regression.score(X_test,y_test)

from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)

# print(model.feature_importances_)

feat_importances = pd.Series(model.feature_importances_, index= X.columns)
feat_importances.nlargest(5).plot(kind='barh')
# plt.show()

#splitting the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.1,random_state=0)
# print(X_test.columns)
# print(X_train.shape)

#model building
from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()

#hyperparameter tuning

n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# print(n_estimator)
max_features = ['auto', 'sqrt']

max_depth = [int(x) for x in np.linspace(5, 30, num=6)]

min_samples_split = [2, 5, 10, 15, 100]

min_samples_leaf = [1, 2, 5, 10]

from sklearn.model_selection import  RandomizedSearchCV

random_grid={
               'n_estimators':n_estimators,
               'max_features':max_features,
               'max_depth':max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf
}

# print(random_grid)
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='neg_mean_squared_error',
                               n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)

rf_random.fit(X_train, y_train )
# print(X_test)
prediction = rf_random.predict(X_test)

# print(prediction)
plt.clf()
# sns.distplot(y_test-prediction)

plt.scatter(y_test, prediction)
# plt.show()

plt.scatter(range(y_test.shape[0]),y_test)
plt.plot(pred,label='linear regression')
plt.plot(prediction, label="random forest regressor")
plt.tight_layout()

# model pickling
import pickle

file = open("random_forest_regressor.pkl", "wb")
pickle.dump(rf_random, file)
