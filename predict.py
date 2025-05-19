import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

# Standardizing numeric data
def clean_and_convert_columns(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '', regex=False)
            .astype(float)
        )
    return df

# Adding more features to the dataset
def add_features(df):
    df['BMI'] = df['weight'] / ((df['height']/ 1000) ** 2)
    df['weight_height_ratio'] = df['weight'] / df['height']
    
    return df

# Load test data
test_df = pd.read_csv('test.csv', usecols=['gender', 'age', 'height', 'weight'])

test_df  =  clean_and_convert_columns(test_df, ['height', 'weight'])

# filling NaNs
for col in ['weight', 'height', 'age']:
    test_df.fillna(test_df[col].median(), inplace=True)

add_features(test_df)

# Defining features available for both datasets
le = LabelEncoder()
test_df['gender_encoded'] = le.fit_transform(test_df['gender'])

# Load trained model
model = joblib.load('model.pkl')

# Select feature columns (adjust based on what was used in training)
feature_cols = ['gender_encoded', 'age', 'height', 'weight', 'BMI', 'weight_height_ratio']

X_test = test_df[feature_cols]

# Predict
predictions = model.predict(X_test)

# Save predictions
output = pd.DataFrame(predictions, columns=['bust_circumference', 'waist_circumference', 'hip_circumference'])
output.to_csv('predictions.csv', index=False)

print("✅ Predictions saved to predictions.csv")