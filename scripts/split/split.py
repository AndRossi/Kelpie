import pandas as pd
import os

# Get the directory of the current script
script_directory = os.path.dirname(os.path.abspath(__file__))
# Extract the directory from the CSV file path
csv_directory = os.path.dirname(script_directory + "/split")

# Combine the script directory with the file name
file_path = os.path.join(script_directory, 'data.csv')

# Check if the file path exists
if os.path.exists(file_path):
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
    validation_txt_path = os.path.join(csv_directory, 'valid.txt')
    test_txt_path = os.path.join(csv_directory, 'test.txt')

    # Save as text files without the "Test" label in the last column
    df[df['Split'] == 'Train'].iloc[:, :-1].to_csv(train_txt_path, sep='\t', index=False)
    df[df['Split'] == 'Validation'].iloc[:, :-1].to_csv(validation_txt_path, sep='\t', index=False)
    df[df['Split'] == 'Test'].iloc[:, :-1].to_csv(test_txt_path, sep='\t', index=False)
else:
    print(f"Error: File not found at {file_path}")