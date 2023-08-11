import argparse
import re
import os

def extract_sequences(input_file, index_file, output_file=None, separate=False, output_dir=None):
    # Read the index numbers from the index file
    with open(index_file, "r") as index_f:
        sample_numbers = set(map(int, index_f.read().split()))

    # Extract sequences with matching sample numbers
    sequences = {}
    with open(input_file, "r") as input_f:
        header = None
        sequence = ""
        for line in input_f:
            line = line.strip()
            if line.startswith(">"):
                # Save the previous sequence and start a new one
                if header is not None and sequence:
                    sample_number_match = re.search(r"(\d+)$", header)
                    if sample_number_match:
                        sample_number = int(sample_number_match.group(1))
                        if sample_number in sample_numbers:
                            sequences[header] = sequence
                header = line[1:]
                sequence = ""
            else:
                sequence += line

        # Save the last sequence
        if header is not None and sequence:
            sample_number_match = re.search(r"(\d+)$", header)
            if sample_number_match:
                sample_number = int(sample_number_match.group(1))
                if sample_number in sample_numbers:
                    sequences[header] = sequence

    # Write the selected sequences to the output file or separate .fasta files
    if separate:
        if output_dir is None:
            output_dir = "./separate_fastas"
        os.makedirs(output_dir, exist_ok=True)
        for header, sequence in sequences.items():
            output_file = os.path.join(output_dir, f"{header.replace('>', '')}.fasta")
            with open(output_file, "w") as out_f:
                out_f.write(f">{header}\n{sequence}\n")
    else:
        if output_file is None:
            output_file = f"{input_file}_new.fasta"
        with open(output_file, "w") as output_f:
            for header, sequence in sequences.items():
                output_f.write(f">{header}\n{sequence}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract sequences from a fasta file based on sample numbers in an index file")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input fasta file")
    parser.add_argument("-n", "--index", type=str, required=True, help="Path to the index txt file")
    parser.add_argument("-o", "--output", type=str, help="Name of the output fasta file")
    parser.add_argument("-s", "--separate", action="store_true", help="Extract each sequence as a separate .fasta file")
    parser.add_argument("-d", "--output_dir", type=str, help="Output directory for separate .fasta files")
    args = parser.parse_args()

    input_file = args.input
    index_file = args.index
    output_file = args.output
    separate = args.separate
    output_dir = args.output_dir

    extract_sequences(input_file, index_file, output_file, separate, output_dir)

