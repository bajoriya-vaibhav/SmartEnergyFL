import pandas as pd
import os

# Relative path to your .feather file
relative_path = './data/train_processed.feather'

# Convert to absolute path
feather_file_path = os.path.abspath(relative_path)

# Read the feather file
df = pd.read_feather(feather_file_path)

# Show the data
print(df.info())
print((pd.to_datetime(df['timestamp']).astype('int64').head(50) % 1_000_000_000//24 ))
# print(df['weekday_hour'].head(20))
# print(df[].groupby(''))
