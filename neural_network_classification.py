import sys
import csv

if len(sys.argv) != 2:
    print("Usage: python neural_network_classification.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
in_data = []
labeled_out = []

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 8:
            continue  # skip incomplete rows
        in_data.append([row[3], row[4], row[5]])
        labeled_out.append([row[6], row[7]])
