import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("data.csv")
 
data = data.dropna(axis = 0)


y=data['price']
housefeatures = ['floors','sqft_living','sqft_lot','bedrooms','bathrooms', 'yr_built']
X = data[housefeatures]

from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

#split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

#Instaniate ridge model
RFG = RandomForestRegressor(n_estimators=100)
RFG.fit(X_train, y_train)


pickle.dump(RFG,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))


