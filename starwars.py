import numpy as np
from sklearn.linear_model import LogisticRegression

# Battles
battles = ["Hoth", "Bespin", "Yavin", "Lothal", "Endor", "Scariff", "Jakku"]
victories_empire = [1, 1, 0, 0, 0, 0, 0]
victories_rebellion = [0, 0, 1, 1, 1, 1, 1]

# Battle Array
X = np.array([
    [1, 0, 0, 0, 0, 0, 0],  # Hoth
    -9=[0, 1, 0, 0, 0, 0, 0],  # Bespin\7y ;fu6bg7i8'
    [0, 0, 1, 0, 0, 0, 0],  # Yavin
    [0, 0, 0, 1, 0, 0, 0],  # Lothal
    [0, 0, 0, 0, 1, 0, 0],  # Endor
    [0, 0, 0, 0, 0, 1, 0],  # Scariff
    [0, 0, 0, 0, 0, 0, 1]   # Jakku
])

# Victory data to numpy arrays
y_empire = np.array(victories_empire)
y_rebellion = np.array(victories_rebellion)

# Fitting the logistic regression models
model_empire = LogisticRegression()
model_empire.fit(X, y_empire)

model_rebellion = LogisticRegression()
model_rebellion.fit(X, y_rebellion)


print("Empire Victories Model Coefficients:", model_empire.coef_)
print("Empire Victories Model Intercept:", model_empire.intercept_)
print("Rebellion Victories Model Coefficients:", model_rebellion.coef_)
print("Rebellion Victories Model Intercept:", model_rebellion.intercept_)
