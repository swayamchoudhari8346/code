# Install required libraries (run in terminal / command prompt):
# pip install pandas matplotlib scikit-learn
#py -3.11 -m pip install pandas matplotlib scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

df = pd.DataFrame({
    'Bedrooms': [1,1,2,2,2,3,3,3,4,4],
    'Rent': [10000,11000,15000,16000,17000,22000,23000,24000,30000,32000]
})
print(df.to_string(index=False))

X, Y = df[['Bedrooms']], df['Rent']
model = DecisionTreeRegressor(criterion='squared_error', max_depth=3, random_state=42)
model.fit(X, Y)

bedrooms = int(input("Enter Number of Bedrooms: "))
rent = model.predict(pd.DataFrame({'Bedrooms': [bedrooms]}))[0]
print("Predicted Monthly Rent: ₹", round(rent, 2))

plt.figure(figsize=(12, 7))
plot_tree(model, feature_names=['Bedrooms'], filled=True, rounded=True)
plt.title("CART Regression Tree")
plt.show()