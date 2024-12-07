import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from collections import defaultdict


train_df = pd.read_csv('train_jumps_processed.csv')
test_df = pd.read_csv('testjumps_processed.csv')


# mapping dict
race_map = {rid: i for i, rid in enumerate(train_df['race_id'].unique())}

train_df['race_id_idx'] = train_df['race_id'].map(race_map)
test_df['race_id_idx'] = test_df['race_id'].map(race_map)

target = 'win'

# training data
y_train = train_df[target].values
print(train_df.columns)

# Drop columns including 'race_id' now that we have 'race_id_idx'
X_train = train_df.drop(columns=['race_id', 'date', 'horse', 'jockey', 'trainer', 'comment', 'win']).values

# testing data
y_test = test_df[target].values
X_test = test_df.drop(columns=['race_id', 'date', 'horse', 'jockey', 'trainer', 'comment', 'win']).values

race_ids = train_df['race_id_idx'].values

# print(X_train)
maxCount = train_df.groupby('race_id_idx').size().max()
# # extract race id's
# race_id_train = X_train[:, -1].astype(int)
# race_id_test = X_test[:, -1].astype(int)
# X_train = X_train[:, :-1]
# X_test = X_test[:, :-1]

# numOfRaces = len(race_map)

# print (maxCount)



races = defaultdict(list)
for features, result, race in zip(X_train, y_train, race_ids):
    print(features, result, race)
    races[race].append((features, result))

features_dim = X_train.shape[1]
all_races_input = []
all_races_target = []

for r in sorted(races.keys()):
    runners = races[r]
    features = [runner[0] for runner in runners]
    target = [runner[1] for runner in runners]

    features = np.array(features)
    target = np.array(target)

    winner = np.argmax(target)

    num_of_runners = target.shape[0]

    if num_of_runners < maxCount:
        for i in range(maxCount - num_of_runners):
            features = np.vstack([features, np.zeros(features_dim)])

    all_races_input.append(features)
    all_races_target.append(winner)

all_races_input = np.array(all_races_input)
all_races_target = np.array(all_races_target)

all_races_input = torch.tensor(all_races_input, dtype=torch.float32)
all_races_target = torch.tensor(all_races_target, dtype=torch.long)

print(all_races_input)
    

class RaceDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

train_dataset = RaceDataset(all_races_input, all_races_target)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)



class SetModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super(SetModel, self).__init__()
        self.shared_mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_final = nn.Linear(hidden_dim, 1)  # later produce logits per horse

    def forward(self, x):
        # x: [batch_size, max_horses, feature_dim]
        # Pass each horse through shared MLP
        batch_size, max_horses, _ = x.size()
        horse_embeds = self.shared_mlp(x) # shape [batch, max_horses, hidden_dim]

        # Aggregate (e.g., mean)
        race_embed = horse_embeds.mean(dim=1, keepdim=True) # [batch, 1, hidden_dim]

        # Combine race_embed back with each horse
        # For simplicity, just add them:
        combined = horse_embeds + race_embed.expand(-1, max_horses, -1)

        # Predict logits for each horse
        # shape: [batch, max_horses, 1]
        logits = self.fc_final(combined).squeeze(-1) # [batch, max_horses]
        return logits

model = SetModel(feature_dim=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

subset_race_ids = test_df['race_id_idx'].unique()[:10]

for epoch in range(30):
    model.train()
    for batch_x, batch_y in train_loader:
        # batch_x: [bs, max_horses, feature_dim]
        # batch_y: [bs] index of the winner horse
        optimizer.zero_grad()
        logits = model(batch_x)  # [bs, max_horses]
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item()}")

    model.eval()
    print("Predictions for the first 10 races:")
    with torch.no_grad():
        for race_id in subset_race_ids:
            # Extract data for this race
            race_mask = test_df['race_id_idx'] == race_id  # Filter rows for the race
            X_race = torch.tensor(X_test[race_mask], dtype=torch.float32)
            race_ids = torch.tensor([race_id] * len(X_race), dtype=torch.long)
            
            # Get model predictions
            outputs = model(X_race, race_ids)
            predicted_winner = torch.argmax(outputs, dim=0).item()  # Winner is the horse with the max score
            
            # Print prediction
            print(f"Race ID: {race_id}, Predicted Winner: {predicted_winner}")
