import pandas as pd


df = pd.read_csv("Sensor_Full_Train_Data_Cleaned_Preprocessed_Normalized.csv")

#Mapping of labels
mapping = {
    "Drink from bottle/cup": 0,
    "Glasses on/off": 0,
    "Pull air toward your face": 0,
    "Pinch knee/leg skin": 0,
    "Scratch knee/leg skin": 0,
    "Write name on leg": 0,
    "Text on phone": 0,
    "Feel around in tray and pull out an object": 0,
    "Write name in air": 0,
    "Wave hello": 0,
    "Above ear - pull hair": 1,
    "Forehead - pull hairline": 2,
    "Forehead - scratch": 3,
    "Eyebrow - pull hair": 4,
    "Eyelash - pull hair": 5,
    "Neck - pinch skin": 6,
    "Neck - scratch": 7,
    "Cheek - pinch skin": 8
}
df['gesture'] = df['gesture'].map(mapping).fillna(df['gesture'])
df.to_csv("Sensor_Full_Train_Data_Cleaned_Preprocessed_Normalized_LabelEncoded_Final.csv", index=False)
print("done")
