import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_iris

dataset = load_iris()
df = pd.DataFrame(dataset.data,columns=dataset.feature_names)

c = df.columns

x = df[[c[0]]]
y = df[c[2]]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

model = LinearRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

plt.scatter(x_test,y_test,color='g')
plt.plot(x_test,y_pred,color='c',label='Predicted regression')
plt.xlabel(c[0])
plt.ylabel(c[2])
plt.title('Iris dataset')
plt.legend()
plt.show()
