import pandas as pd

# Load your CSV file, specifying common NA indicators
df = pd.read_csv('SensorTrainDataCleaned2.csv', na_values=['', ' ', 'NA', 'NaN'])

# Replace missing values only for numeric columns
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Save the cleaned DataFrame back to CSV
df.to_csv('sensordataimputed3.csv', index=False)

print("done")
