import pandas as pd


data = pd.read_csv('2005-2024.csv')
columns_to_drop = ['region', 'course_id', 'off', 'race_name', 'dist', 'dist_f', 'ran', 'num', 'btn', 'wgt', 'horse_id', 'time', 'sp', 'jockey_id', 'trainer_id', 'prize', 'sire_id', 'dam_id', 'damsire_id']
data = data.drop(columns=columns_to_drop, axis=1)
# print(data)
# print(data)
# one_hot = ['type', 'sex', 'hg']
# one_hot_encoded_data = pd.get_dummies(data, columns=one_hot)
# one_hot_encoded_data.to_csv('one_hot_encoded_sample.csv', index=False)

data.to_csv('cleaned.csv', index=False)