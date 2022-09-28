import sys
#!{sys.executable} -m pip install shap==0.38.1

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
#from sklearn.model_selection import KFold
#from sklearn.datasets import load_digits


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix


from sklearn.metrics import classification_report

data = pd.read_csv('https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv')
#df_test = data[data['Make'].isnull()].copy()
#pd.to_csv(df_test,"test_set.csv")

data_all = data.copy()

def preprocessing_dataset(df_main):

    df_main.drop(['Meter Id', 'Marked Time','VIN', 'Issue time', 'Ticket number', 'Location', 'Route', 'Violation Description'], axis = 1, inplace=True)
    df_main['Issue Date'] = pd.to_datetime(df_main['Issue Date'])
    df_main['issue_yearmonth'] = (df_main['Issue Date'].dt.year)*100+df_main['Issue Date'].dt.month
    
      #df_main['issue_year'] = df_main['Issue Date'].dt.year

      #bins_iss = np.linspace(df_main['issue_year'].min(),df_main['issue_year'].max(),4)
      #df_main['issue_status'] = pd.cut(df_main['issue_year'], bins_iss, labels=['old_issue','current_issue','new_issue'])


    df_main['expiry_year'] = df_main['Plate Expiry Date']//100
 #   bins = [df_main['expiry_year'].min(),df_main['expiry_year'].quantile(0.33), df_main['expiry_year'].quantile(0.66), df_main['expiry_year'].max()]
 #   df_main['expiry_status'] = pd.cut(df_main['expiry_year'], bins, labels=['old','current','new'])
 #   df_main['expiry_status'] = df_main['expiry_status'].astype("category")


      #categoryVariableList = ['RP State Plate', 'Body Style', 'Color', 'Agency', 'Violation code']
      #cols_missing_values = ['RP State Plate', 'Make', 'Body Style', 'Color', 'Agency', 'Violation code', 'Fine amount', 'Latitude', 'Longitude']

      #no_missing_values = df_main.isnull().sum()
      #for col in cols_missing_values:
      #  if no_missing_values[col] ==0:
      #    break;
      #  elif no_missing_values[col] >0 and col in categoryVariableList:
      #    df_main[col] = df_main[col].apply(lambda x: x.fillna(x.value_counts().index[0]))
      #  elif no_missing_values[col] >0 and col not in categoryVariableList:
      #    df_main[col].fillna(df_main[col].median(), inplace = True)
   # df_main['expiry_year'].fillna(df_main['expiry_year'].median(), inplace = True)

    df_main.drop(['Issue Date', 'Plate Expiry Date'], axis = 1, inplace=True)

    #col_lst = ['issue_yearmonth','RP State Plate', 'Body Style', 'Color', 'Agency', 'Violation code', 'Fine amount', 'Latitude', 'Longitude']
    #df_main.dropna(axis = 0, subset = col_lst, how = 'any', inplace = True)
    
    numericalVariableList = ['issue_yearmonth', 'expiry_year','Fine amount', 'Latitude', 'Longitude']
    categoryVariableList = ['RP State Plate', 'Body Style', 'Color', 'Agency', 'Violation code']

    for var in numericalVariableList:

        df_main[var].fillna(df_main[var].median(),inplace=True)

        
    for var in categoryVariableList:
        mode = df_main[var].mode()[0]
        df_main[var].fillna(mode,inplace=True)
        
        df_main[var] = df_main[var].astype("category")
               
        # frequency encoding
        new_var = var +'_count'
        df_main[new_var] = df_main[var].map(df_main[var].value_counts())
        df_main.drop([var], axis = 1, inplace=True)
    
      # One hot encoding
      #  df_main = pd.get_dummies(df_main, columns=categoryVariableList)
        
    # PCA for dimensionality reduction
    

    return df_main


df_train_all = preprocessing_dataset(data_all)

df_train = df_train_all[df_train_all['Make'].notnull()].copy()
df_test = df_train_all[df_train_all['Make'].isnull()].copy()

print(df_train['Make'].value_counts().head(25))
cut_off = df_train['Make'].value_counts().head(25)[-1]
df_train['counts'] = df_train.Make.map(df_train.Make.value_counts())
y_data = df_train['counts']
y_data = np.where(y_data < cut_off, 0, 1) # freq < cut_off category 0, popular '1'
#actual_fails = (y_data ==0).sum()
#actual_success = (y_data ==1).sum()
#print(actual_fails, actual_success)
X_data = df_train.copy()
X_data.drop(['Make', 'counts'], axis = 1, inplace=True)

scaler = preprocessing.StandardScaler().fit(X_data)
X_scaled = scaler.transform(X_data)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X_scaled, y_data, test_size=0.3, random_state=1)
log_reg = LogisticRegression()
log_reg.fit(X_train1, y_train1)
y_pred1 = log_reg.predict(X_test1)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log_reg.score(X_test1, y_test1)))

print(metrics.accuracy_score(y_pred1,y_test1))

print(classification_report(y_test1, y_pred1))

log_reg = LogisticRegression()
log_reg.fit(X_data, y_data)

df_test.drop(['Make'], axis = 1, inplace=True)
X_test = df_test.copy()

#scaler = preprocessing.StandardScaler().fit(X_test)
X_test_scaled = scaler.transform(X_test)

y_pred_prob = log_reg.predict_proba(X_test_scaled)[:,1]


filename = 'prediction_model.sav'
pickle.dump(log_reg, open(filename, 'wb'))
pickle.dump(scaler, open('scaling_data.sav', 'wb'))

# pd.DataFrame(y_pred_prob).to_csv("y_prediction_probabilities.csv")

