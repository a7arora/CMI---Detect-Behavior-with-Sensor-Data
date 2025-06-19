import pandas as pd

# Load your CSV file
df = pd.read_csv("SensorTrainDataCleaned11.csv")
exclude_columns = ['gesture', 'behavior', 'phase', 'orientation', 'subject', 'sequence_counter', 'row_id', 'sequence_type', 'row_id', 'sequence_id']
columns_of_interest = df.columns.difference(exclude_columns)
summary_stats = df[columns_of_interest].agg(['mean', 'std'])
summary_stats.to_csv('sensormeansandstds11.csv', index=True)
