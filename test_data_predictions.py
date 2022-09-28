import pickle
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing

df_test_sample = pd.DataFrame()
data = pd.DataFrame()

try: 
    f = open('user_input.json')
    input = json.load(f)
    df_test_sample = pd.DataFrame(input, index=['i',])
except IOError:
    print ("Could not read input json file:")

# load the model from disk
filename = 'prediction_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))
#Scaler = pickle.load('scaling_data.sav', 'rb')

data = pd.read_csv('https://s3-us-west-2.amazonaws.com/pcadsassessment/parking_citations.corrupted.csv')

df_test = data[data['Make'].isnull()].copy()

df_test.append(df_test_sample, ignore_index=False, verify_integrity=False, sort=None) 
#pd.to_csv(df_test,"test_set.csv")

def preprocessing_testset(df_main):

    df_main.drop(['Meter Id', 'Marked Time','VIN', 'Issue time', 'Ticket number', 'Location', 'Route', 'Violation Description'], axis = 1, inplace=True)
    df_main['Issue Date'] = pd.to_datetime(df_main['Issue Date'])
    df_main['issue_yearmonth'] = (df_main['Issue Date'].dt.year)*100+df_main['Issue Date'].dt.month

    df_main['expiry_year'] = df_main['Plate Expiry Date']//100


    df_main.drop(['Issue Date', 'Plate Expiry Date'], axis = 1, inplace=True)

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
        
    df_main.drop(['Make'], axis = 1, inplace=True)
    
    scaler = preprocessing.StandardScaler().fit(df_main)
    
    X_test_scaled = scaler.transform(df_main)

    return X_test_scaled
    
X_test = preprocessing_testset(df_test)
y_pred = loaded_model.predict_proba(X_test)[:,1]
prediction = y_pred[-1]

print("output probability for given sample")
print(prediction)

with open("output.json", "w") as outfile:
    outfile.write(str(prediction))
