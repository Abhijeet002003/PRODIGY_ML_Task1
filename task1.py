import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the Kaggle dataset into pandas DataFrames
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Separate features (X) and target variable (y) for training data
X_train = train_df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y_train = train_df['SalePrice']

# Split the training data into training and validation sets (80% training, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Evaluate the model on the validation set
mse_val = mean_squared_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)

print(f'Validation Mean Squared Error: {mse_val}')
print(f'Validation R-squared: {r2_val}')

# Once you are satisfied with the model's performance, you can make predictions on the test set.
# Extract features (X_test) from the test data and make predictions
X_test = test_df[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y_test_pred = model.predict(X_test)

# Save the predictions for submission (e.g., to a CSV file)
submission_df = pd.DataFrame({'Id': test_df['Id'], 'SalePrice': y_test_pred})
submission_df.to_csv('submission.csv', index=False)
