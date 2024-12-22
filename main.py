import os
from dotenv import load_dotenv
import pymongo
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import OneHotEncoder

load_dotenv()

mongo_client = pymongo.MongoClient(os.getenv("DATABASE_URL"))
db = mongo_client["eviks"]
posts = db["posts"]
post_list = posts.find({}, {"price": 1, "sqm": 1, "rooms": 1, "floor": 1, "totalFloors": 1, "apartmentType": 1})

df = pd.DataFrame(post_list)

categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
encoder = OneHotEncoder(sparse_output=False)
one_hot_encoded = encoder.fit_transform(df[categorical_columns])
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
final_df = pd.concat([df.drop(categorical_columns, axis=1), one_hot_df], axis=1)
data = final_df.drop(['_id'], axis=1)

x = data.drop(['price'], axis=1)
y = data["price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 0)    

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

print(f"intercept: {linear_model.intercept_}")
print(f"slope: {linear_model.coef_}")

y_pred = linear_model.predict(x_test)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"mape: {mape}")

y_final = linear_model.predict(np.array([[5, 5, 2, 40, 0, 1]]))
print(f"predicted response: {y_final}")

print(data.head())

correlation_matrix = data.corr()
sns.heatmap(correlation_matrix,
            cmap = "BrBG",
            fmt = '.2f',
            linewidths = 2,
            annot = True)
plt.title('Feature Correlation Matrix')
plt.show()