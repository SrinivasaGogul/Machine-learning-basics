from regressor import LinearRegression
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


matplotlib.use('TkAgg')

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 1000)

data = pd.read_csv("../datasets/HousingData.csv")


data["CRIM"] = data["CRIM"].fillna(data["CRIM"].mean())
data["ZN"] = data["ZN"].fillna(data["ZN"].mean())
data["INDUS"] = data["INDUS"].fillna(data["INDUS"].mean())
data["CHAS"] = data["CHAS"].fillna(data["CHAS"].mean())
data["AGE"] = data["AGE"].fillna(data["AGE"].mean())
data["LSTAT"] = data["LSTAT"].fillna(data["LSTAT"].mean())

print(data.isnull().sum())
X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LinearRegression(0.0001, 15)

model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
print(model.weights)
print(model.bias)
mse = mean_squared_error(Y_test, y_pred)
r2_scr = r2_score(Y_test, y_pred)

print(mse)
print(r2_scr)


