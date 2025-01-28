import csv
import random

def random_sample(input_file, output_file, sample_size):
    with open(input_file, 'r') as csv_file:
        # Read the CSV file
        reader = csv.reader(csv_file)
        header = next(reader)  # Assuming the first row is a header

        # Read all lines into a list
        all_lines = list(reader)

        # Ensure the requested sample size is not greater than the total number of lines
        sample_size = min(sample_size, len(all_lines))

        # Randomly select sample_size lines
        selected_lines = random.sample(all_lines, sample_size)

    with open(output_file, 'w', newline='') as output_csv:
        # Write the header to the output file
        csv.writer(output_csv).writerow(header)

        # Write the selected lines to the output file
        csv.writer(output_csv).writerows(selected_lines)

if __name__ == "__main__":
    input_file_path = "kelpie_sufficient_conve_antique.csv"  # Specify your input CSV file
    output_file_path = "kelpie_sufficient_conve_antique_sampled.csv"  # Specify the desired output CSV file
    sample_size = 199  # Specify the number of lines to randomly select

    random_sample(input_file_path, output_file_path, sample_size)