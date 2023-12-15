import pandas as pd

def create_csv_with_random_correct_predictions(filtered_ranks_path, output_path, sample_size=60):
    '''
    Creates a CSV file with a random sample of correct tail predictions from the filtered_ranks file.

    :param filtered_ranks_path: Path to the filtered_ranks file.
    :param output_path: Path where the output CSV file will be saved.
    :param sample_size: Number of correct tail predictions to sample. Default is 50.
    '''
    try:
        # Reading the filtered ranks file
        data = pd.read_csv(filtered_ranks_path, sep=';', header=None, names=['head', 'relation', 'tail', 'head_rank', 'tail_rank'], encoding='latin1')

# Filtering rows where the tail prediction is correct (tail_rank == 1)
        correct_tail_predictions = data[data['tail_rank'] == 1]

        # Sampling the specified number of correct tail predictions
        sampled_predictions = correct_tail_predictions.sample(n=sample_size, random_state=42)

        # Writing the head, relation, tail to a new CSV file
        sampled_predictions[['head', 'relation', 'tail']].to_csv(output_path, index=False, sep='\t')

        print(f"File with sampled correct tail predictions saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Paths to the filtered_ranks file and output file (replace with actual paths)
filtered_ranks_path = './filtered_ranks.csv'
output_path = './correct_tail_predictions.csv'

# Running the function
create_csv_with_random_correct_predictions(filtered_ranks_path, output_path)