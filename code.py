import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv("dataSet/student-scores.csv")

# Calculate TOTAL_MARK as the average of all subject scores
df['TOTAL_MARK'] = (df['math_score'] + df['history_score'] + df['physics_score'] + 
                    df['chemistry_score'] + df['biology_score'] + df['english_score'] + 
                    df['geography_score']) / 7.0

# Drop individual subject score columns
df = df.drop(columns=['math_score', 'history_score', 'physics_score', 'chemistry_score', 
                      'biology_score', 'english_score', 'geography_score'])

# Encode categorical features
labelencoder = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    labelencoder[column] = le

# Define features (X) and target (y)
X = df.drop(columns='TOTAL_MARK')
y = df['TOTAL_MARK']

# ================================
# LinearRegression Model
# ================================
linear_model = LinearRegression()
linear_model.fit(X, y)

# ================================
# RandomForestRegressor Model
# ================================
rf_model = RandomForestRegressor()
rf_model.fit(X, y)

# ================================
# DecisionTreeRegressor Model
# ================================
dt_model = DecisionTreeRegressor()
dt_model.fit(X, y)

# ================================
# Support Vector Regressor (SVR) Model
# ================================
svr_model = SVR(kernel='rbf')  # 'rbf' kernel is commonly used for non-linear regression
svr_model.fit(X, y)

# ================================
# Save the trained models and LabelEncoders
# ================================
joblib.dump(linear_model, 'linear_regression_model.pkl')
joblib.dump(rf_model, 'random_forest_regressor_model.pkl')
joblib.dump(dt_model, 'decision_tree_regressor_model.pkl')
joblib.dump(svr_model, 'svr_model.pkl')

# Save the LabelEncoders
joblib.dump(labelencoder, 'label_encoders.pkl')

# Model Performance (optional)
def evaluate_model(model, model_name):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"{model_name} MSE: {mse}, R²: {r2}")

print("\n===== Model Performance =====")
evaluate_model(linear_model, "Linear Regression")
evaluate_model(rf_model, "Random Forest Regressor")
evaluate_model(dt_model, "Decision Tree Regressor")
evaluate_model(svr_model, "SVR")
