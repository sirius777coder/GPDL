import argparse
from Bio import SeqIO

def parse_args():
    parser = argparse.ArgumentParser(description="Rename sequences in a fasta file.")
    parser.add_argument("-i", "--input", required=True, help="Input fasta file.")
    parser.add_argument("-o", "--output", help="Output fasta file. If not provided, input file will be used.")
    parser.add_argument("-p", "--prefix", help="Prefix for renaming. Default is the original fasta name.")

    return parser.parse_args()

def main():
    args = parse_args()

    sequence_file = args.input
    output_file = args.output if args.output else sequence_file

    renamed_sequences = []
    sample_number = 1

    with open(sequence_file, "r") as file:
        for record in SeqIO.parse(file, "fasta"):
            header_parts = record.description.split(", ")
            new_prefix = args.prefix if args.prefix else sequence_file.split("/")[-1].split(".")[0]
            new_header = f"{new_prefix}_{sample_number}"
            renamed_header = f">{new_header}"
            record.id = renamed_header
            record.description = ""
            renamed_sequences.append(record)
            sample_number += 1

    with open(output_file, "w") as file:
        SeqIO.write(renamed_sequences, file, "fasta-2line")

    print("Sequences renamed and saved successfully.")

if __name__ == "__main__":
    main()
