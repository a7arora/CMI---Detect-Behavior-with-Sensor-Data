import pandas as pd
import numpy as np
import csv

EXCLUDE_COLUMNS = [
    'row_id', 'sequence_id', 'sequence_type', 'sequence_counter', 
    'subject', 'orientation', 'behavior', 'phase', 'gesture', 'missing'
]
FLOAT_PRECISION = 3
df = pd.read_csv('SensorTrainDataClean2.csv')
for col in df.select_dtypes(include=['object']):
    df[col] = df[col].astype(str).str.replace(r'[^\x20-\x7E]', '', regex=True)
sensor_columns = [
    col for col in df.columns 
    if col not in EXCLUDE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
]
for col in sensor_columns:
    series = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
    series = series.interpolate(method='linear', limit_direction='both')
    df[col] = series
df[sensor_columns] = df[sensor_columns].apply(pd.to_numeric, errors='coerce')
df[sensor_columns] = df[sensor_columns].fillna(df[sensor_columns].mean())
float_format_string = f'%.{FLOAT_PRECISION}f'
df.to_csv(
    'SensorTrainDataCleaned11.csv',
    index=False,
    encoding='utf-8-sig',
    float_format=float_format_string,
    quoting=csv.QUOTE_NONE
)

print(f"✅ Cleaning complete. CSV saved with {FLOAT_PRECISION} decimal places.")

