import pandas
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import numpy as np 
import matplotlib.pyplot as plt
#i really do hope i did this right not gonna lie 


''' 
The following is the starting code for path1 for data reading to make your first step easier.
'dataset_1' is the clean data for path1.
'''
dataset_1 = pandas.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
dataset_1['Brooklyn Bridge']      = pandas.to_numeric(dataset_1['Brooklyn Bridge'].replace(',','', regex=True))
dataset_1['Manhattan Bridge']     = pandas.to_numeric(dataset_1['Manhattan Bridge'].replace(',','', regex=True))
dataset_1['Queensboro Bridge']    = pandas.to_numeric(dataset_1['Queensboro Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Williamsburg Bridge']  = pandas.to_numeric(dataset_1['Williamsburg Bridge'].replace(',','', regex=True))
dataset_1['Total']  = pandas.to_numeric(dataset_1['Total'].replace(',','', regex=True))
dataset_1['High Temp']      = pandas.to_numeric(dataset_1['High Temp'].replace(',','', regex=True))
dataset_1['Low Temp']     = pandas.to_numeric(dataset_1['Low Temp'].replace(',','', regex=True))
dataset_1['Precipitation']    = pandas.to_numeric(dataset_1['Precipitation'].replace(',','', regex=True))

BMQ_x = pandas.concat([dataset_1['Brooklyn Bridge'], dataset_1['Manhattan Bridge'],dataset_1['Queensboro Bridge'] ], axis=1)
BMW_x = pandas.concat([dataset_1['Brooklyn Bridge'], dataset_1['Manhattan Bridge'],dataset_1['Williamsburg Bridge']], axis=1)
BQW_x = pandas.concat([dataset_1['Brooklyn Bridge'], dataset_1['Queensboro Bridge'],dataset_1['Williamsburg Bridge']], axis=1)
MQW_x = pandas.concat([dataset_1['Manhattan Bridge'] , dataset_1['Queensboro Bridge'],dataset_1['Williamsburg Bridge']], axis=1)
total_y = dataset_1['Total']

TempRain_concat = pandas.concat([dataset_1['High Temp'] , dataset_1['Low Temp'],dataset_1['Precipitation']], axis=1)

# print(dataset_1.to_string()) #This line will print out your data

"""The use of the code provided is optional, feel free to add your own code to read the dataset. The use (or lack of use) of this code is optional and will not affect your grade."""

def BridgeRegression(bridgedata,total_y):
    
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(bridgedata,total_y)


    coef = regr.coef_
    intercept = regr.intercept_
    
    y_pred_test = regr.predict(bridgedata)


    bridge_r2 = r2_score(total_y, y_pred_test)
    bridge_mse = mean_squared_error(total_y, y_pred_test)
    
    
    plt.figure(figsize = (15,10))
    plt.scatter(total_y,y_pred_test)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted')
   
    print(coef,intercept,"\n")
    
    return bridge_r2,coef,intercept,bridge_mse

def tempRainCount(tempraindata,total_y):
    
    
    regr = linear_model.LinearRegression(fit_intercept=True)
    regr.fit(tempraindata,total_y)


    coef = regr.coef_
    intercept = regr.intercept_
    
    y_pred_test = regr.predict(tempraindata)
   

    tempraincheck_r2 = r2_score(total_y, y_pred_test)
    tempraincheck_mse = mean_squared_error(total_y, y_pred_test)
   
    return tempraincheck_r2,coef,intercept,tempraincheck_mse
    
def isitRaining(tempraindata,total_y):
    
    X = total_y.values
    y = tempraindata.values
    y_01 = []
    
    
    

    # 0.01 raining is still raining so we need all values greater than 0 to be turned into one
    for i in y:
        if i > 0:
            y_01.append(1)
        else:
            y_01.append(0)
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_01, test_size=0.4, random_state=1)
    gnb = GaussianNB()
    
    X_train = np.asarray(X_train)
    X_test = np.asarray(X_test)
    y_train = np.asarray(y_train)
    X_train= X_train.reshape(-1, 1)
    X_test= X_test.reshape(-1, 1)
    y_train= y_train.ravel()
    
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    
    print(y_pred)
    print("GNB Model Accuracy in percent:",metrics.accuracy_score(y_test,y_pred) * 100 )
    
    pass
    
#main
print("~~~~~~~~~~~")
print("Problem #1\n")
print("Brooklyn, Manhattan, Queensboro Bridges; Coef(s) + Intercept:")
BMQ = BridgeRegression(BMQ_x,total_y)
print("Brooklyn, Manhattan, Queensboro Bridges; r2 Value + MSE:")
print(BMQ[0],BMQ[3])
print("~~~~~~~~~~~")

print("Brooklyn, Manhattan, Williamsburg Bridges; Coef(s) + Intercept:")
BMW = BridgeRegression(BMW_x,total_y)
print("Brooklyn, Manhattan, Williamsburg Bridges; r2 Value + MSE:")
print(BMW[0],BMW[3])
print("~~~~~~~~~~~")

print("Brooklyn, Queensboro, Williamsburg Bridges; Coef(s) + Intercept:")
BQW = BridgeRegression(BQW_x,total_y)
print("Brooklyn, Queensboro, Williamsburg Bridges; r2 Value + MSE:")
print(BQW[0],BQW[3])
print("~~~~~~~~~~~")

print("Manhattan, Queensboro, Williamsburg Bridges; Coef(s) + Intercept:")
MQW = BridgeRegression(MQW_x,total_y)
print("Manhattan, Queensboro, Williamsburg Bridges; r2 Value + MSE:")
print(MQW[0],MQW[3])
print("~~~~~~~~~~~")



if BMQ[0] > (BMW[0] or BQW[0] or MQW[0]):
    print("Sensors should be installed along the Brooklyn Bridge, Manhattan Bridge, and Queensboro Bridge\n")
    print("Equation that describes ideal traffic model:\nTotal =",BMQ[1][0],"* (Brookyln Bridge) +",BMQ[1][1],"* (Manhattan Bridge) +",BMQ[1][2],"* (Queensboro Bridge)+",BMQ[2])
elif BMW[0] > (BMQ[0] or BQW[0] or MQW[0]):
    print("Sensors should be installed along the Brooklyn Bridge, Manhattan Bridge, and Williamsburg Bridge\n")
    print("Equation that describes ideal traffic model:\nTotal =",BMW[1][0],"* (Brookyln Bridge) +",BMW[1][1],"* (Manhattan Bridge) +",BMW[1][2],"* (Williamsburg Bridge) +",BMW[2])
elif BQW[0] > (BMW[0] or BMQ[0] or MQW[0]):
    print("Sensors should be installed along the Brooklyn Bridge, Queensboro Bridge, and Williamsburg Bridge\n")
    print("Equation that describes ideal traffic model:\nTotal =",BQW[1][0],"* (Brookyln Bridge) +",BQW[1][1],"* (Queensboro Bridge) +",BQW[1][2],"* (Williamsburg Bridge) +",BQW[2])
elif MQW[0] > (BMW[0] or BMQ[0] or BQW[0]):
    print("Sensors should be installed along the Manhattan Bridge, Queensboro Bridge, and Williamsburg Bridge\n")
    print("Equation that describes ideal traffic model:\nTotal =",MQW[1][0],"* (Manhattan Bridge) +",MQW[1][1],"* (Queensboro Bridge) +",MQW[1][2],"* (Williamsburg Bridge) +",MQW[2])

print("~~~~~~~~~~~")
print("Problem #2\n")


TempRainCheck = tempRainCount(TempRain_concat,total_y)
print("Equation that describes weather to cyclyist model:\nTotal =",TempRainCheck[1][0],"* (High Temp) +",TempRainCheck[1][1],"* (Low Temp) +",TempRainCheck[1][2],"* (Precipitation) +",TempRainCheck[2])
print("\nHigh Temp, Low Temp, Precipitation; r2 Value + MSE:")
print(TempRainCheck[0],TempRainCheck[3])
print("~~~~~~~~~~~")
print("Problem #3\n")
print("Zeroes represent no rain, Ones represent rain")

raining = isitRaining(dataset_1['Precipitation'],total_y)

