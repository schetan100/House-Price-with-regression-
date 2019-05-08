# -*- coding: utf-8 -*-
"""
Created on Thu May  2 11:00:14 2019

@author: Joker
"""
#libraries 
import operator 

# data processing
import pandas as pd

# linear algebra
import numpy as np 

#matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style
import matplotlib.pyplot as pt
%matplotlib inline


# data visualization
import seaborn as sns


# Algorithms
from sklearn.model_selection import train_test_split,GridSearchCV,KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier 
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from scipy.stats import skew,norm
from scipy.stats.stats import pearsonr
from sklearn.linear_model import Lasso,ElasticNet,BayesianRidge, LassoLarsIC
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
#import xgboost as xgb
from scipy import stats
#import lightgbm as lgb

#Importing Data

test_df = pd.read_csv("F:/Practice Data/House Price/test.csv")

train_df = pd.read_csv("F:/Practice Data/House Price/train.csv")
#train = pd.read_csv("F:/Practice Data/House Price/train.csv")


#save the ID column
train_ID = train_df['Id']
test_ID = test_df['Id']

#Drop the ID column since it is unnecessary for the prediction process
train_df.drop("Id",axis =1,inplace = True)
test_df.drop("Id",axis =1,inplace= True)

print ("Train data: \n")
print ("Number of columns: " + str (train_df.shape[1]))
print ("number of rows: " + str (train_df.shape[0]))

print('\nTest data: \n')
print ("number of columns:" + str (test_df.shape[1]))
print ("Number of columns:" +  str (test_df.shape[0]))

#Data Exploration/Analysis

train_df.info()

#Describe
train_df.describe()

#Data Header 
train_df.head(8)
#test_df.head(8)

#ScatterPlot for GrLivArea Outliers
fig, ax = plt.subplots()
ax.scatter(x = train_df['GrLivArea'], y = train_df['SalePrice'], color = 'blue')
plt.ylabel('SalePrice', fontsize = 13)
plt.xlabel('GrLivArea', fontsize = 13)
plt.show()

train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 200000)].index, axis = 0, inplace = True)


#Check the graphic again
fig, ax = plt.subplots()
ax.scatter(train_df['GrLivArea'], train_df['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
plt.show()
#train_test = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)]
#train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)].index, axis = 0, inplace = True)

#GarageCars
plt.scatter(x=train_df["GarageCars"], y =np.log(train_df.SalePrice))
plt.show()

#GarageArea
plt.scatter(x=train_df["GarageArea"], y =np.log(train_df.SalePrice))
plt.show()

#Elimintate Outliers in GarageArea
train_df = train_df[train_df.GarageArea <1200]
print (train_df)
plt.scatter(x=train_df.GarageArea,y=np.log(train_df.SalePrice))
plt.show()

#TotalBsmtSF
var = 'TotalBsmtSF'
data = pd.concat([train_df['SalePrice'],train_df[var]],axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim =0.800000);
plt.show()

#scatter plot LotArea/salePrice
var = 'LotArea'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
data.plot.scatter(x= var, y='SalePrice', ylim =(0,800000));
plt.show();

#test['SalePrice'] = np.log(train_df.SalePrice)

#Categorical variables

#OverallQual Variables 
train_df.OverallQual.unique()
quality_pivot=train_df.pivot_table(index="OverallQual",values="SalePrice",aggfunc=np.median)
print (quality_pivot)
quality_pivot.plot(kind="bar",color="blue")
plt.xticks(rotation=0)
plt.show()

#box plot overallqual/salePrice
var = 'OverallQual'
data = pd.concat([train_df['SalePrice'], train_df[var]], axis=1)
f, ax =plt.subplots(figsize=(8,6))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.show();

#year built
var  = 'YearBuilt'
data= pd.concat([train_df['SalePrice'], train_df[var]], axis =1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x=var, y='SalePrice', data=data)
fig.axis(ymin=0, ymax=800000)
plt.xticks(rotation=90);
plt.show();

#Check Normalization on SalesPrice Variable

sns.distplot(train_df['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()

#We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
train_df["SalePrice"] = np.log1p(train_df["SalePrice"])

#Check the new distribution 
sns.distplot(train_df['SalePrice'] , fit=norm);

# Get the fitted parameters used by the function
(mu, sigma) = norm.fit(train_df['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

#Now plot the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')

#Get also the QQ-plot
fig = plt.figure()
res = stats.probplot(train_df['SalePrice'], plot=plt)
plt.show()

#Numeric Features
#numeric_features = train_df.select_dtypes(include = [np.number])
#numeric_features.dtypes

#concatenate the train and test data in the same dataframe

ntrain = train_df.shape[0]
ntest = test_df.shape[0]
y_train = train_df.SalePrice.values
all_data = pd.concat((train_df, test_df)).reset_index(drop=True)
all_data.drop(['SalePrice'], axis=1, inplace=True)
print("all_data size is : {}".format(all_data.shape))


#Missing Values Treatment 

#Missing Data in Train
missing_train = train_df.isnull().sum()/len(train_df.index)*100
missing_train = missing_train[missing_train > 0]
missing_train.sort_values(inplace=True)
plt.xlabel("Column Name")
plt.ylabel("Percentage Missing")
plt.title("Percentage Missing in Train Data")
len(missing_train)

#Missing Data in Train
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_train.index, y =missing_train)
plt.xlabel("Column Name")
plt.ylabel("Percentage Missing")
plt.title("Percentage Missing in Train Data")

#Missing Data in Test
missing_test = test_df.isnull().sum()/len(test_df.index)*100
missing_test = missing_test[missing_test >0]
missing_test.sort_values(inplace=True)

f, ax = plt.subplots(figsize= (15,6))
plt.xticks(rotation='90')
sns.barplot(x= missing_test.index, y = missing_test)
plt.xlabel("Column Name")
plt.ylabel("Percentage Missing")
plt.title("Percentage Missing in Test Data")

len(missing_test)

#Missing Data Ration in Both Data Set 

all_data_na = (all_data.isnull().sum()/ len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending = False)
# [:30]
missing_data =pd.DataFrame({'Missing Ratio':all_data_na})
missing_data.head(20)

f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


#Correlation Matrix to measure linear relationship with variable 
correlation_matrix = train_df.corr().round(2)
# annot = True to print the values inside the square
print(correlation_matrix['SalePrice'].sort_values(ascending =False))
f,ax = plt.subplots(figsize= (30,9))
sns.heatmap(data=correlation_matrix, annot=True)

#Top 10 Variables
cols = correlation_matrix.nlargest(10,'SalePrice')['SalePrice'].index
cm = np.corrcoef(train_df[cols].values.T)
sns.set(font_scale= 1.25)
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt ='.2f', annot_kws={'size': 10}, yticklabels = cols.values,
                xticklabels = cols.values)
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
plt.show()

#Finding Object in DataSet
#data_X.dtypes.value_counts()
#data_X.select_dtypes(include = [object]).columns

#Convert NA into None
for col in ('PoolQC','MiscFeature','GarageType','Alley','Fence','FireplaceQu','GarageFinish',
           'GarageQual','GarageCond','MasVnrType','MSSubClass'):
    all_data[col] = all_data[col].fillna('None')
    
#Convert NAN into 0 
for col in ('GarageYrBlt','GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)

#missing values are likely zero for no basement 
for col in ('BsmtFinSF1','BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF','BsmtFullBath',
            'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
    
#
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)

#for below categorical basement-related feature NaN means that there is no basement 
for col in ('BsmtQual', 'BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
    
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

#Neighborhood
sns.stripplot(x= train_df.Neighborhood, y = target, 
              order = np.sort(train_df.Neighborhood.unique()),
              jitter = 0.1)
plt.xticks(rotation=45)

Neighborhood_meanSP = \
    train_df.groupby('Neighborhood')['SalePrice'].mean()
 
Neighborhood_meanSP = Neighborhood_meanSP.sort_values()

sns.pointplot(x = train_df.Neighborhood.values, y = train_df.SalePrice,
              order = Neighborhood_meanSP.index)
 
plt.xticks(rotation=45)

#msZoning classification: 'RL' is common
all_data ['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

#functional: NA is typical
all_data["Functional"] = all_data["Functional"].fillna('Typ')

#Electrical
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

#KitchenQual
all_data['KitchenQual'] =all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

#Extrerior !st and Exterior 2nd
all_data ['Exterior1st']= all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd']= all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

#sale type
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
    
#dropping as same value 'AllPub' for all records except 2NA and 1 'NoSeWa'
all_data = all_data.drop(['Utilities'], axis=1)

#check if any missing value present
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head()


#Transforming required numerical features to categorical 
all_data['MSSubClass']= all_data['MSSubClass'].apply(str)
all_data['OverallCond'] =all_data['OverallCond'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)


#Label Encoding some categorical variables
#for information in their ordering set

from sklearn.preprocessing import LabelEncoder

cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold')
#apply LabelEncoder to categorical features
for c in cols:
    lbl = LabelEncoder()
    lbl.fit(list(all_data[c].values))
    all_data[c] = lbl.transform(list(all_data[c].values))
#shape
print('Shape all_data: {}'.format(all_data.shape))

#add total surface area as TotalSf = basement + firstflr + secondflr
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


#log transform skewed numeric features 

numeric_features = all_data.dtypes[all_data.dtypes != "object"].index

skewed_features = all_data[numeric_features].apply(lambda x : skew(x.dropna())).sort_values(ascending=False)
#compute skewness
print ("\skew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_features})   
skewness.head(7)

#Box cox transformation of highly skewed features
skewness = skewness[abs(skewness) > 0.75]
print ("There are {} skewed numerical features to box cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p 
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
                                             
#Dummy Categorical

all_data = pd.get_dummies(all_data)
print(all_data.shape)


train_df = pd.DataFrame(all_data[:ntrain])
test_df = pd.DataFrame(all_data[ntrain:])


#Linear regression Modeling
#Lasso Regression
#Gradient Boosting Regression

#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train_df.values)
    rmse= np.sqrt(-cross_val_score(model, train_df.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)
    

#Base Modelling 

#Lasso Regressino 
lasso = make_pipeline(RobustScaler(), Lasso(alpha = 0.0005, random_state = 1))

#Gradient Boosting Regression :
GBoost = GradientBoostingRegressor(loss='huber', learning_rate=0.05, n_estimators=3000,
                                   min_samples_split=10, min_samples_leaf=15,max_depth=4,
                                   random_state=5,max_features='sqrt')

#Elastic Net Regression
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))

#Kernel Ridge Regression :
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)


#XGBoost 
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

#LightGBM

model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)


#Score from Models
score = rmsle_cv(lasso)
print ("\n Lasso score: {:.4f} ({:.4f})\n".format(score.mean(),score.std()))

score = rmsle_cv(GBoost)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(ENet)
print("ElasticNet score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = rmsle_cv(KRR)
print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#Stacking Models

class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)  
    
#Average base model score 
averaged_models = AveragingModels(models = (ENet, GBoost, KRR, lasso))

score = rmsle_cv(averaged_models)
print(" Averaged base models score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

#defining RMSLE evaluation function
def RMSLE (y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))


averaged_models.fit(train_df.values, y_train) 
stacked_train_pred = averaged_models.predict(train_df.values)
stacked_pred = np.expm1(averaged_models.predict(test_df.values))
print("RMSLE score on the train data:") 
print(RMSLE(y_train,stacked_train_pred))
print("Accuracy score:") 
averaged_models.score(train_df.values, y_train)


ensemble = stacked_pred *1
submit = pd.DataFrame()
submit['id'] = test_ID
submit['SalePrice'] = ensemble
submit.to_csv('F:/Practice Data/House Price/Final.csv',encoding='utf-8', index = False)
submit.head()                                






