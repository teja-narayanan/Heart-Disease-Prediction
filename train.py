import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model 

# import the df
# df = pd.read_csv('heart_data.csv')
df = pd.read_csv('inputs\heart_data.csv')

df.head()

# drop unnamed field
df = df.drop("Unnamed: 0", axis=1)


# plot the features
sns.lmplot(x="biking", y="heart.disease", data=df)
sns.lmplot(x="smoking", y="heart.disease", data=df)


# split features and target
X = df.drop('heart.disease', axis=1)
y = df['heart.disease']


# train and test models
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# train linear regression model
model = linear_model.LinearRegression()


# fit the model and print score
model.fit(X_train, y_train)
print(model.score(X_train, y_train))


# predict the model
prediction_test = model.predict(X_test)

# print(y_test, prediction_test)
print("Mean sq. error between y_test and predicted =", np.mean(prediction_test-y_test)**2)


# let's save and pickle our trained model
import pickle
pickle.dump(model, open('model.pkl', 'wb'))


# for testing purposes - load the pickled file
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[70.1, 26.3]]))