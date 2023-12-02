import pandas as pd
import os

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Extract the directory from the CSV file path
csv_directory = os.path.dirname(script_directory+"/SAMPLE")

# Combine the script directory with the file name
file_path = os.path.join(script_directory, 'data.csv')



# Load CSV file
df = pd.read_csv(file_path)

# Split ratios
train_ratio = 0.7
validation_ratio = 0.15
test_ratio = 0.15

# Create 'Split' column
df['Split'] = pd.cut(
    df.index,
    bins=[0, train_ratio * len(df), (train_ratio + validation_ratio) * len(df), len(df)],
    labels=['Train', 'Validation', 'Test']
)

# Same directory as the data.csv file
train_txt_path = os.path.join(csv_directory, 'train.txt')
validation_txt_path = os.path.join(csv_directory, 'validation.txt')
test_txt_path = os.path.join(csv_directory, 'test.txt')

# Save as text files
df[df['Split'] == 'Train'].to_csv(train_txt_path, sep='\t', index=False)
df[df['Split'] == 'Validation'].to_csv(validation_txt_path, sep='\t', index=False)
df[df['Split'] == 'Test'].to_csv(test_txt_path, sep='\t', index=False)
