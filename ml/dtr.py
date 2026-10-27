import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

df = pd.DataFrame({
    'Bedrooms': [1,1,2,2,2,3,3,3,4,4],
    'Rent': [10000,11000,15000,16000,17000,22000,23000,24000,30000,32000]
})

X, Y = df[['Bedrooms']], df['Rent']
model = DecisionTreeRegressor(max_depth=3, random_state=42)
model.fit(X, Y)

bedrooms = int(input("Enter Number of Bedrooms: "))
rent = model.predict([[bedrooms]])[0]
print("Predicted Monthly Rent: ₹", round(rent, 2))

# Simple graph
plot_tree(model, feature_names=['Bedrooms'], filled=True)
plt.show()
