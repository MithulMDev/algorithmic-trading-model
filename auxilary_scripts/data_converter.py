import csv

# Specify the input and output file paths
input_file = 'E:\\trading_algo_model\\organized_files_data\\ACC_F1.txt\\nifty-banknifty-intraday-data-train_2017_AUG_01AUG_01AUG_ACC_F1.txt'  # Input text file (adjust path if needed)
output_file = 'output.csv'  # Output CSV file

# Open the text file to read
with open(input_file, 'r') as infile:
    lines = infile.readlines()  # Read all lines into a list

# Open the output CSV file to write
with open(output_file, 'w', newline='') as outfile:
    csv_writer = csv.writer(outfile)

    # Optionally, write the header row (if known)
    header = ["Type", "Date", "Time", "open", "high", "low", "close", "volume"]
    csv_writer.writerow(header)

    # Process each line and write it to the CSV
    for line in lines:
        # Split the line by commas, removing any unwanted whitespace
        row = line.strip().split(',')
        
        # Check if the row has the expected number of columns
        if len(row) == 8:
            csv_writer.writerow(row)
        else:
            print(f"Skipping invalid row: {line.strip()}")

print(f"Data successfully written to {output_file}")
