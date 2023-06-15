############################################
#*********Import Basic libraries*********
#############################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


############################################
#*********Getting the dataset*********
#############################################
tmpDF = pd.read_csv("Regression restaurant bill.csv")
print(tmpDF)


############################################
#********Gathering some information*********
############################################
print(tmpDF.head())

print(tmpDF.tail())

print(tmpDF.shape)

print(tmpDF.columns)

print(tmpDF.info())

print(tmpDF.isnull().sum()/tmpDF.shape[0]*100)

print(tmpDF.isnull().sum())


############################################
#**********Taking all the inputs***********
############################################

target_col_name="total_bill"

drop_col_nm=['None']

dateTimeColsNames=['none']

drop_nan_col_bound=24

num_col_nm_strip=['total_bill', 'tip', 'size']

num_col_nm_strip_2=['sex', 'smoker', 'day', 'time']

rank_col=['None']

columns_to_split=['none']

delimiters=['none']

makedecision="none"


cl=['sex', 'time', 'day', 'smoker']
num_col_nm_strip_2.clear()
num_col_nm_strip_2.extend(cl)

df=tmpDF


#######################################################
#*********************** Encoding ********************
#######################################################

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder

allCatCol=list(df.select_dtypes(include="object").columns)

categorical_column=None

categorical_column=['sex', 'time', 'day', 'smoker']
df["sex"]=df["sex"].map({'male': 0, 'female': 1})

df["time"]=df["time"].map({'lunch': 0, 'dinner': 1})


df["smoker"]=df["smoker"].map({'no': 0, 'yes': 1})

encoding= OneHotEncoder(sparse=False, drop="first")
#*********** Performing one-Hot encoding ************
result_encod=encoding.fit_transform(df[['day']])
dataFrame_encode_col=pd.DataFrame(result_encod,columns=['day_sat', 'day_thu', 'day_tue', 'day_wed'])
numeric_col=df[['total_bill', 'sex', 'time', 'tip', 'size', 'smoker']]
final_dataset=pd.merge(dataFrame_encode_col,numeric_col,left_index=True,right_index=True)


#######################################################
#**** Separating dependent and independent feature ****
#######################################################

x=final_dataset.drop(target_col_name, axis=1)
y=final_dataset[target_col_name]


#######################################################
#****** Performing VIF for multicollinearity **********
#######################################################

from statsmodels.stats.outliers_influence import variance_inflation_factor

#** Calculate the VIF for each feature in the training data **
vif_x = pd.DataFrame()
vif_x["VIF Factor"] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif_x["features"] = x.columns

#**** Print the VIF scores for the training data ****
print(vif_x)

#** Identify the features with high VIF scores in the training data **
high_vif_train = vif_x[vif_x["VIF Factor"] > 10]["features"]
print(high_vif_train)

#** Drop the highly correlated features from data **
x = x.drop(high_vif_train, axis=1)


############################################################################
#**************** Separating Data into train and test data *****************
############################################################################

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=1)
#Change the random_state parameter value. It can change the accuracy


############################################################################
#****************************** model training *****************************
############################################################################

from sklearn import metrics
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor

xgb = XGBRegressor()
xgb.fit(x_train,y_train)
xgbtd=xgb.predict(x_test)
print('r2 score',r2_score(y_test,xgbtd))


######################################
#****** Model evaluation ************
######################################

# make predictions on the testing set
y_pred = xgb.predict(x_test)

# calculate the evaluation metrics

#Mean Squared error
mse = mean_squared_error(y_test, y_pred)
print(mse)

#Root mean squared error
rmse = np.sqrt(mse)
print(rmse)

#r2_score
r2 = r2_score(y_test, y_pred)
print(r2)

mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(mae)

Mean absolute percentage error
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(mape)


############################################
#********** Plot the residuals *************
############################################

residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.show()


