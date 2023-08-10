#Sequence Recovery calculating script
#Written by @Immortals on August 9th, 2023
#A first try on pathlib and typing module. Remember to modify the other filtering script later.

import argparse
import logging
import re
from pathlib import Path
from typing import List
import typing as T

import torch
import biotite
import biotite.sequence.io.fasta as fasta

root_path = Path("/dssg/home/acct-clschf/clschf/zzq/ADS/filter/recovery")
PathLike = T.Union[str, Path]

'''
Calculating sequence recovery between a reference sequence and a set of sequences, and filter them out, optionally.
The input format can be either a multiple-sequences fasta file or a folder containing many single fasta files.

Usage:
[Required]
"-i", "--input": Input fasta file or folder.
"-o", "--output": The name of result file.
"-
[Optional]
"-o", "--output": The name of result file, using this flag is strongly recommended. Default = "recovery.txt"
"-s", "--seperate": This flag is used when input is a folder.
"-t", "--threshold": Sequence recovery threshold. Sequences above this value would be kept, otherwise filtered out.
'''

def calculate_sequence_recovery(s1, s2):
    AA_alphabet = "ACDEFGHIKLMNPQRSTVWY-"
    aa_to_index = {aa: index for index, aa in enumerate(AA_alphabet)}

    s1_indices = [aa_to_index[aa] for aa in s1]
    s2_indices = [aa_to_index[aa] for aa in s2]

    s1_one_hot = torch.nn.functional.one_hot(torch.tensor(s1_indices), len(AA_alphabet))
    s2_one_hot = torch.nn.functional.one_hot(torch.tensor(s2_indices), len(AA_alphabet))

    seq_recovery_rate = torch.sum(torch.sum(s1_one_hot * s2_one_hot, axis=-1)) / len(s1)

    return seq_recovery_rate.item()


def natural_sort_key(s: Path) -> List:
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s.name)
    ]

def create_parser():
    parser = argparse.ArgumentParser(description="Calculating Sequence Recovery")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Path to the input FASTA file or folder",
    )
    parser.add_argument(
        "-o", "--output", type=Path, 
        default= 'recovery.txt',
        help="Output file name"
    )
    parser.add_argument(
        "-s",
        "--separate",
        action="store_true",
        help="Flag to indicate if input is a folder with separate sequences",
    )
    parser.add_argument(
        "-r",
        "--reference",
        type=Path,
        required=True,
        help="Path to the reference sequence file",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, help="Sequence recovery threshold"
    )
    return parser


def run(args):
    input_path = args.input.resolve()
    output_file = args.output.resolve()
    separate_flag = args.separate
    reference_file = args.reference.resolve()
    recovery_threshold = args.threshold

    refseq = fasta.get_sequence(fasta.FastaFile.read(reference_file))
    results = []
    sample_numbers = []

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%d/%m/%y %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Calculating sequence recovery to {reference_file}......")

    if separate_flag:
        logger.info(f"Loading separate sequences from {input_path}......")
        fasta_files = [
            fasta_path
            for fasta_path in input_path.glob("*.fasta")
            if fasta_path.is_file()
        ]
        fasta_files += [
            fasta_path for fasta_path in input_path.glob("*.fa") if fasta_path.is_file()
        ]

        for fasta_path in sorted((fasta_files), key=natural_sort_key):
            single_seq = fasta.get_sequence(fasta.FastaFile.read(fasta_path))
            sample_match = re.search(r"(\d+)",(fasta_path.stem))
            if sample_match:
                sample_number = int(sample_match.group(1))
                sample_numbers.append(sample_number)
            else:
                logger.warning(
                    f"Could not find the sample number of pdb file. You might need to double check the file format."
                )
                pass
            seq_recovery = calculate_sequence_recovery(refseq, single_seq)
            logger.info(f"Sample: {sample_number}, Sequence recovery = {seq_recovery}.")
            results.append((sample_number, seq_recovery))
    else:
        logger.info(f"Loading sequences from {input_path}......")
        multiseq = fasta.FastaFile.read(str(input_path))
        for header, sequence in fasta.get_sequences(multiseq).items():
            sample_match = re.search(r"(\d+)", header)
            if sample_match:
                sample_number = int(sample_match.group(1))
                sample_numbers.append(sample_number)
            else:
                logger.warning(
                    f"Could not find the sample number of pdb file. You might need to double check the file format."
                )
            seq_recovery = calculate_sequence_recovery(refseq, sequence)
            logger.info(f"Sample: {sample_number}, Sequence recovery = {seq_recovery}.")
            results.append((sample_number, seq_recovery))

    with output_file.open("w") as out:
        for sample_number, seq_recovery in results:
            out.write(f"{sample_number}\t{seq_recovery:.3f}\n")

        if args.threshold:
            filtered_numbers = [
                sample_number
                for sample_number, seq_recovery in results
                if seq_recovery >= recovery_threshold
            ]
            logger.info(
                f"Number of Seqs kept after filtering: {len(filtered_numbers)}."
            )
            filter_id_file = root_path / f"filterID_1.txt"
            with filter_id_file.open("w") as f:
                f.write("\n".join(map(str, sorted(filtered_numbers))))


def main():
    parser = create_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
