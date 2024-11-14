import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress


df = pd.read_csv('train_data.csv', header=0)

def get_position(position):
    try:
        return int(position)
    except:
        return np.nan


df['position'] = df['pos'].apply(get_position)
df['win'] = df['position'].apply(lambda x: 1 if x == 1 else 0)

quantitative_cols = ['age', 'lbs', 'or', 'rpr', 'ts', 'dist_m', 'dec', 'secs']


# These must be numbers, or it would cause a potential model to corrupt
for column in quantitative_cols:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Feature Engineering

# Counts how many races the said horse has ran in up to this date
df['num_of_races'] = df.groupby(df['horse']).cumcount()

df['num_of_wins'] = df.groupby('horse')['win'].transform(lambda row: row.shift().cumsum().fillna(0).astype(int))

df['win_rate'] = (df['num_of_wins'] / df['num_of_races']).fillna(0)

df['average_position'] = df.groupby(df['horse'])['position'].transform(lambda row: row.shift().expanding().mean().fillna(0).astype(float))

df['average_rpr'] = df.groupby(df['horse'])['rpr'].transform(lambda row: row.shift().expanding().mean().fillna(0).astype(float))

df['runs_on_ground'] = df.groupby(['horse', 'going']).cumcount()

df['wins_on_ground'] = df.groupby(['horse', 'going'])['win'].transform(lambda row: row.shift().cumsum().fillna(0).astype(int))


df['win_rate_on_ground'] = (df['wins_on_ground'] / df['runs_on_ground']).fillna(0)


distance_bins = [0, 1000, 1600, 2400, 3200, 4000, 4800, 5600, 6400, np.inf]
distance_labels = ['Sprint', 'Mile', 'Middle', 'Classic', 'Long', 'Extended', 'Marathon', 'Ultra', 'Extreme']

df['distance_category'] = pd.cut(df['dist_m'], bins=distance_bins, labels=distance_labels)


df['num_of_races_at_distance'] = df.groupby(['horse', 'distance_category']).cumcount()

df['num_of_wins_at_distance'] = df.groupby(['horse', 'distance_category'])['win'].transform(lambda row: row.shift().cumsum().fillna(0).astype(int))


df['win_rate_at_distance'] = (df['num_of_wins_at_distance'] / df['num_of_races_at_distance']).fillna(0)



df['jockey_num_of_races'] = df.groupby('jockey').cumcount()


df['jockey_num_of_wins'] = df.groupby('jockey')['win'].transform(lambda row: row.shift().cumsum().fillna(0).astype(int))


df['jockey_win_rate'] = (df['jockey_num_of_wins'] / df['jockey_num_of_races']).fillna(0)


df['trainer_num_of_races'] = df.groupby('trainer').cumcount()


df['trainer_num_of_wins'] = df.groupby('trainer')['win'].transform(lambda row: row.shift().cumsum().fillna(0).astype(int))


df['trainer_win_rate'] = (df['trainer_num_of_wins'] / df['trainer_num_of_races']).fillna(0)


categorical_cols = ['course', 'going', 'surface', 'class', 'type', 'age_band', 'sex', 'distance_category']
# One-hot encode the categorical variables
df = pd.get_dummies(df, columns=categorical_cols, dtype=int)

# Define the window size
window_size = 3  # Last 3 races

# Rolling features for each horse
df['rolling_avg_position'] = df.groupby('horse')['position'].transform(
    lambda x: x.shift().rolling(window=window_size, min_periods=1).mean()
)

df['rolling_avg_rpr'] = df.groupby('horse')['rpr'].transform(
    lambda x: x.shift().rolling(window=window_size, min_periods=1).mean()
)

df['rolling_sum_wins'] = df.groupby('horse')['win'].transform(
    lambda x: x.shift().rolling(window=window_size, min_periods=1).sum()
)

# Convert 'date' to datetime if not already done
df['date'] = pd.to_datetime(df['date'])

# Calculate days since last race
df['days_since_last_race'] = df.groupby('horse')['date'].diff().dt.days

# Fill NaN values (e.g., first race for a horse) - High value to indicate that the horse has not raced in a while
df['days_since_last_race'] = df['days_since_last_race'].fillna(365)

def calculate_trend(x):
    x = x.dropna()
    if len(x) >= 2:
        y = x.values
        x_vals = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = linregress(x_vals, y)
        return slope
    else:
        return 0

# Calculate trend over the last N races
df['rpr_trend'] = df.groupby('horse')['rpr'].transform(
    lambda x: x.shift().rolling(window=window_size, min_periods=2).apply(calculate_trend, raw=False)
)

# Fill NaN values
df['rpr_trend'] = df['rpr_trend'].fillna(0)

# Function to calculate the current streak
def calculate_streak(wins):
    streak = []
    current_streak = 0
    for win in wins.shift():  # Shift to exclude current race
        if pd.isnull(win):
            streak.append(0)
        elif win == 1:
            current_streak += 1
            streak.append(current_streak)
        else:
            current_streak = 0
            streak.append(current_streak)
    return pd.Series(streak, index=wins.index)

# Apply to the dataframe
df['win_streak'] = df.groupby('horse')['win'].transform(calculate_streak)

# Exponentially weighted moving average of RPR
df['ewm_rpr'] = df.groupby('horse')['rpr'].transform(
    lambda x: x.shift().ewm(span=window_size, adjust=False).mean()
)

# Fill NaN values
df['ewm_rpr'] = df['ewm_rpr'].fillna(method='bfill')

# Time decay factor based on days since last race
decay_factor = 0.9  # Adjust as needed

def weighted_cumsum(values, times):
    weights = decay_factor ** (times)
    return np.cumsum(values * weights) / np.cumsum(weights)

# Calculate weighted win rate
df['weighted_win_rate'] = df.groupby('horse').apply(
    lambda group: weighted_cumsum(group['win'].shift().fillna(0), group['days_since_last_race'].shift().fillna(0))
).reset_index(level=0, drop=True)

# Fill NaN values
df['weighted_win_rate'] = df['weighted_win_rate'].fillna(0)

def calculate_form_score(group):
    """
    Calculate a weighted form score based off of last 5 results of the horse.
    Most recent has the highest weight with it decreasing for each race further back.
    """
    def score_last_5_races(positions):
        # Convert positions to list and take last 5 races
        pos_list = list(positions)[-5:]
        print(pos_list)
        
        # Weights for each position (most recent first)
        weights = [1.0, 0.8, 0.6, 0.4, 0.2][:len(pos_list)]
        
        # Calculate score for each position
        score = 0
        for position, weight in zip(pos_list, weights):
            # Better positions (e.g., 1st) get higher scores
            if pd.notna(position):
                score += (1 / position) * weight
            else:
                # Horse not finished, give it a poor rating
                score += (1/10) * weight
            
        return score

    # Shift to exclude current race and calculate scores
    return group.shift().rolling(window=5, min_periods=1).apply(score_last_5_races)

# Apply to dataframe
df['form_score'] = df.groupby('horse')['position'].transform(calculate_form_score)

# Horse-Jockey Interaction Win Rate
df['horse_jockey_win_rate'] = df.groupby(['horse', 'jockey'])['win'].transform(
    lambda x: x.shift().cumsum() / (x.shift().expanding().count())
).fillna(0)

# Fill missing numerical values with median
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# Cap outliers using the 1st and 99th percentiles
for col in numerical_cols:
    lower_bound = df[col].quantile(0.01)
    upper_bound = df[col].quantile(0.99)
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])


# Target encoding for high-cardinality variables
high_cardinality_cols = ['sire', 'dam', 'damsire', 'owner']

for col in high_cardinality_cols:
    # Calculate mean target value for each category
    target_mean = df.groupby(col)['win'].mean()
    df[col + '_te'] = df[col].map(target_mean)
    df[col + '_te'] = df[col + '_te'].fillna(df['win'].mean())
    df.drop(columns=[col], inplace=True)


df.drop(columns=['pos', 'position', 'ovr_btn', 'hg', 'dec', 'draw'], inplace=True)




# List of numerical features to scale (interim at this stage, more may be created at a later stage)
numerical_features = ['age', 'lbs', 'or', 'rpr', 'ts', 'dist_m', 'secs',
                      'num_of_races', 'num_of_wins', 'win_rate', 'average_position', 'average_rpr',
                      'runs_on_ground', 'wins_on_ground', 'win_rate_on_ground',
                      'num_of_races_at_distance', 'num_of_wins_at_distance', 'win_rate_at_distance',
                      'jockey_num_of_races', 'jockey_num_of_wins', 'jockey_win_rate',
                      'trainer_num_of_races', 'trainer_num_of_wins', 'trainer_win_rate',
                      'rolling_avg_position', 'rolling_avg_rpr', 'rolling_sum_wins',
                      'days_since_last_race', 'rpr_trend', 'win_streak', 'ewm_rpr',
                      'weighted_win_rate', 'horse_jockey_win_rate',
                      'sire_te', 'dam_te', 'damsire_te', 'owner_te']

# Initialize scaler
scaler = StandardScaler()

# # Fit and transform the numerical features
df[numerical_features] = scaler.fit_transform(df[numerical_features])

constant_columns = [col for col in df.columns if df[col].nunique() == 1]
print(constant_columns)
df.drop(columns=constant_columns, inplace=True)
df.drop(columns=['rpr', 'ts', 'race_id', 'pattern', 'rating_band', 'sex_rest', 'dist_m'], inplace=True)
df.drop(columns=[
    'date',
    'secs',              # Often highly correlated with distance and can be noisy
    'or',               # Official rating - redundant with RPR which we already used for features
    'age',              # Already captured in age_band dummy variables
    'lbs',              # Weight carried - often less predictive than class/rating
    
    # Redundant cumulative stats (since we have better weighted/rolling versions)
    'num_of_races',
    'num_of_wins',
    'average_position',
    'runs_on_ground',
    'wins_on_ground',
    'num_of_races_at_distance',
    'num_of_wins_at_distance',
    
    # Raw count features that are less useful than their derived rates
    'jockey_num_of_races',
    'jockey_num_of_wins',
    'trainer_num_of_races',
    'trainer_num_of_wins',
    
    # Keep only the most sophisticated versions of features
    'rolling_sum_wins',  # redundant with weighted_win_rate
    'win_rate',         # replaced by weighted_win_rate
    'win_rate_on_ground',  # potentially noisy with small sample sizes
    'win_rate_at_distance', # potentially noisy with small sample sizes
    'dam_te',
    'owner_te'
], inplace=True)



df.to_csv('train_processed.csv', index=False)

print(df)

