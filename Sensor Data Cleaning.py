import pandas as pd
df = pd.read_csv('SensorTrainData.csv')
df['Missing'] = (df == -1).sum(axis=1)
df.to_csv('SensorTrainDataClean2.csv', index=False)