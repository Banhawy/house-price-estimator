import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error
from sklearn.externals import joblib

# Load the data set
df = pd.read_csv("ml_house_data_set.csv")

# Remove the fields from the data set that we don't want to include in our model
del df['house_number']
del df['unit_number']
del df['street_name']
del df['zip_code']
# Replace categorical data with one-hot encoded data
features_df = pd.get_dummies(df, columns=['garage_type', 'city'])

# Remove the sale price from the feature data
del features_df['sale_price']

# Create the X and y arrays
X = features_df.as_matrix()
y = df['sale_price'].as_matrix()

# Split the data set in a training set (70%) and a test set (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Fit regression model
model = ensemble.GradientBoostingRegressor(
    n_estimators=1000, # How many decision trees to build, higher numbers = more accuracy = more time to build
    learning_rate=0.1, # How much each additional tree influeces the overall prediction, lower rates = more accuracy
    max_depth=6, # How many layers deep eahc decision tree can be
    min_samples_leaf=9, # How many times a value must appear in a setfor a decision tree to make a decision based on it
    max_features= 0.1, # precentage of features in model chosen at random to consider each time decision tree creates a branch
    loss='huber', # function calculates model's error rate as it learns
    random_state=0
)

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file so we can use it in other programs
joblib.dump(model, 'trained_house_classifier_model.pkl')

# Find the error rate on the training set
mse = mean_absolute_error(y_train, model.predict(X_train))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set
mse = mean_absolute_error(y_test, model.predict(X_test))
print("Test Set Mean Absolute Error: %.4f" % mse)