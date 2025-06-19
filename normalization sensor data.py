import pandas as pd

df = pd.read_csv("SensorTrainDataCleaned11.csv")

exclude_columns = ['gesture', 'behavior', 'phase', 'orientation', 'subject', 'sequence_counter', 'row_id', 'sequence_type', 'sequence_id']

columns_of_interest = df.columns.difference(exclude_columns)

summary_stats = df[columns_of_interest].agg(['mean', 'std'])

print(summary_stats)
print(summary_stats.index)  # This should print: Index(['mean', 'std'], dtype='object')

# Normalize using .loc
df_normalized = df.copy()

for col in columns_of_interest:
    mean_val = summary_stats.loc['mean', col]
    std_val = summary_stats.loc['std', col]
    if std_val != 0:
        df_normalized[col] = (df[col] - mean_val) / std_val
    else:
        df_normalized[col] = 0

# Added float_format to reduce file size
df_normalized.to_csv("SensorTrainDataCleaned_Normalized15.csv", index=False, float_format="%.3f")
