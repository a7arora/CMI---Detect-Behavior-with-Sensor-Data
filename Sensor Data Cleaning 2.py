import pandas as pd
import numpy as np
import csv

# CONFIGURATION
EXCLUDE_COLUMNS = [
    'row_id', 'sequence_id', 'sequence_type', 'sequence_counter', 
    'subject', 'orientation', 'behavior', 'phase', 'gesture', 'missing'
]
FLOAT_PRECISION = 3  # e.g. keep 3 decimal places

# === STEP 1: Load data ===
df = pd.read_csv('SensorTrainDataClean2.csv')

# === STEP 2: Clean invisible characters (for safety) ===
for col in df.select_dtypes(include=['object']):
    df[col] = df[col].astype(str).str.replace(r'[^\x20-\x7E]', '', regex=True)

# === STEP 3: Identify numeric sensor columns ===
sensor_columns = [
    col for col in df.columns 
    if col not in EXCLUDE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
]

# === STEP 4: Interpolation + cleanup ===
for col in sensor_columns:
    # Coerce to numeric (replace non-numeric with NaN, treat -1 as missing)
    series = pd.to_numeric(df[col], errors='coerce').replace(-1, np.nan)
    # Interpolate both directions
    series = series.interpolate(method='linear', limit_direction='both')
    # Assign cleaned series back
    df[col] = series

# === STEP 5: Force all sensor columns back to strict float ===
df[sensor_columns] = df[sensor_columns].apply(pd.to_numeric, errors='coerce')

# === STEP 6: Replace any remaining NaNs with column means ===
df[sensor_columns] = df[sensor_columns].fillna(df[sensor_columns].mean())

# === STEP 7: Export with controlled precision ===
float_format_string = f'%.{FLOAT_PRECISION}f'
df.to_csv(
    'SensorTrainDataCleaned11.csv',
    index=False,
    encoding='utf-8-sig',
    float_format=float_format_string,
    quoting=csv.QUOTE_NONE
)

print(f"✅ Cleaning complete. CSV saved with {FLOAT_PRECISION} decimal places.")

