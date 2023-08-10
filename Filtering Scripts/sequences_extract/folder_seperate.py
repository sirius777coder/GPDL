import os
import re
import argparse
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description="Seperate fasta files into multiple folders.")
    parser.add_argument("-i", "--input", required=True, help="Input folder containing original fasta files.")
    parser.add_argument("-o", "--output", required=True, help="Output folder to store processed folders.")
    parser.add_argument("-t", "--total", type=int, default=10000, help="Range of sample_number. Default is 10000.")
    parser.add_argument("-n", "--number", type=int, required=True, help="Number of separating operations.")

    return parser.parse_args()

# Helper function for natural sorting of file names
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def main():
    args = parse_args()
    input_folder = args.input
    output_folder = args.output
    total_samples = args.total
    num_folders = args.number

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    def get_sample_number(file_name):
        return int(re.search(r"_(\d+)\.fasta", file_name).group(1))

    fasta_files = [f for f in os.listdir(input_folder) if f.endswith(".fasta")]

    filtered_fasta_files = [fasta_file for fasta_file in fasta_files if 1 <= get_sample_number(fasta_file) <= total_samples]

    filtered_fasta_files.sort(key=natural_sort_key)

    for folder_number in range(1, num_folders + 1):
        folder_name = os.path.join(output_folder, f"fasta_r2_{folder_number}")
        os.makedirs(folder_name)

        folder_start = (folder_number - 1) * total_samples // num_folders + 1
        folder_end = folder_number * total_samples // num_folders

        for fasta_file in filtered_fasta_files:
            sample_number = get_sample_number(fasta_file)

            if folder_start <= sample_number <= folder_end:
                src_path = os.path.join(input_folder, fasta_file)
                dst_path = os.path.join(folder_name, fasta_file)

                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)

if __name__ == "__main__":
    main()
