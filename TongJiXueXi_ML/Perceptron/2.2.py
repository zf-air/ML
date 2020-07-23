from sklearn.linear_model import Perceptron
import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([[3,3],[4,3],[1,1]])
y = np.array([1,1,-1])

model = Perceptron()
model.fit(x_train,y)

# print('w:',model.coef_ , '\n','b:',model.intercept_ , '\n')
print("w:", model.coef_, "\nb:", model.intercept_, "\n")
result = model.predict(x_train)

print(result)