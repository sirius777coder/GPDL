from pathlib import Path
import sys
import os
import os.path
import logging
import typing as T
import argparse

import torch
import numpy as np
import esm
from esm.data import read_fasta
from timeit import default_timer as timer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%y/%m/%d %H:%M:%S",
)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


PathLike = T.Union[str, Path]

def create_batched_sequence_datasest(
    sequences: T.List[T.Tuple[str, str]], max_tokens_per_batch: int = 1024
) -> T.Generator[T.Tuple[T.List[str], T.List[str]], None, None]:

    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences:
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pre_sequence', type=str,default=None,
        help=
        """
        where the pre_sequence is stored
        """
    )
    parser.add_argument(
        '--reference', type=lambda x: os.path.expanduser(str(x)),
        help=
        """
        reference pdb
        """
    )
    parser.add_argument(
        '--output_dir', type=lambda x: os.path.expanduser(str(x)),
        help=
        """
        The output directory to write the output pdb files. 
        If the directory does not exist, we just create it. 
        The output file name follows its unique identifier in the 
        rows of the input fasta file"
        """
    )
    parser.add_argument(
        '--final_des_dir', type=lambda x: os.path.expanduser(str(x)),
        help=
        """
        The output directory to write the final designs. 
        If the directory does not exist, we just create it. 
        The output file name follows its unique identifier in the 
        rows of the input fasta file"
        """
    )
    parser.add_argument(
        '--bb_suffix', type=int,
        help=
        """
        The index of start backbone 
        """
    ) 
    parser.add_argument(
        '--step', type=int,
        help=
        """
        The number of rounds for the optimization 
        """
    )
    parser.add_argument(
        "--loss",
        type=int,
        default=None,
        help="rmsd weight",
    )
    parser.add_argument(
        "--t1",
        type=float,
        default=None,
        help="0.01 or 1",
    )
    parser.add_argument(
        "--t2",
        type=int,
        default=None,
        help="1000 or 500",
    )
    parser.add_argument(
        "--max_mut",
        type=int,
        default=None,
        help="percentage of scaffold substituted in the initial roundtrip",
    )
    parser.add_argument(
        "--number",
        type=int,
        default=None,
        help="design num",
    )
    parser.add_argument(
        "--mask_len",
        type=str,
        default=None,
        help="25,25",
    )
    parser.add_argument(
        "--motif_id",
        type=str,
        default=None,
        help="P254-277",
    )
    parser.add_argument(
        "--atom",
        type=str,
        default=None,
        help="N,CA,C,O",
    )
    parser.add_argument(
        "--num-recycles",
        type=int,
        default=None,
        help="Number of recycles to run. Defaults to number used in training (4).",
    )
    parser.add_argument(
        "--max-tokens-per-batch",
        type=int,
        default=8,#128
        help="Maximum number of tokens per gpu forward-pass. This will group shorter sequences together "
        "for batched prediction. Lowering this can help with out of memory issues, if these occur on "
        "short sequences.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=64,
        help="Chunks axial attention computation to reduce memory usage from O(L^2) to O(L). "
        "Equivalent to running a for loop over chunks of of each dimension. Lower values will "
        "result in lower memory usage at the cost of speed. Recommended values: 128, 64, 32. "
        "Default: None."
    )
    parser.add_argument("--cpu-only", help="CPU only", action="store_true")
    parser.add_argument(
        "--cpu-offload", help="Enable CPU offloading", action="store_true"
    )
    args = parser.parse_args()
    return args


# def main(model,output_fasta, num, motif):
def main(model,all_sequences, num, motif):
    args = get_args()
    atoms = (args.atom).split(',')
    # all_sequences = sorted(read_fasta(output_fasta), key=lambda header_seq: len(header_seq[1]))
    # logger.info(f"Loaded {len(all_sequences)} sequences from {output_fasta}")

    # logger.info("Starting Predictions")
    batched_sequences = create_batched_sequence_datasest(
        all_sequences, args.max_tokens_per_batch
    )

    num_completed = 0
    num_sequences = len(all_sequences)
    for headers, sequences in batched_sequences:
        start = timer()
        try:
            output = model.infer(sequences, num_recycles=args.num_recycles)
        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                if len(sequences) > 1:
                    logger.info(
                        f"Failed (CUDA out of memory) to predict batch of size {len(sequences)}. "
                        "Try lowering `--max-tokens-per-batch`."
                    )
                else:
                    logger.info(
                        f"Failed (CUDA out of memory) on sequence {headers[0]} of length {len(sequences[0])}."
                    )

                continue
            raise

        output = {key: value.cpu() for key, value in output.items()}
        pdbs = model.output_to_pdb(output)
        tottime = timer() - start
        time_string = f"{tottime / len(headers):0.1f}s"
        if len(sequences) > 1:
            time_string = time_string + f" (amortized, batch size {len(sequences)})"
        coord = np.empty(shape=(0,3), dtype = float) 
        for header, seq, pdb_string, mean_plddt, ptm, pae in zip(
            headers, sequences, pdbs, output["mean_plddt"], output["ptm"], output['predicted_aligned_error']
        ):
            output_file = Path(f"{args.output_dir}/{num}_{header}.pdb")
            # output_file.write_text(pdb_string)
            num_completed += 1
            mean_pae = torch.mean(pae)
            logger.info(
                f"Predicted structure for {header} with length {len(seq)}"
                f", pLDDT {mean_plddt:0.1f}, "
                f"pTM {ptm:0.3f}, "
                f"pAE {mean_pae:0.1f} in {time_string}."
                f"{num_completed} / {num_sequences} completed."
            )

            plddts = [] #every res
            plddt = {}
            pdb_lines = pdb_string.split("\n")
            for line in pdb_lines:
                line = line.split()
                
                if line==[] or line[0] != 'ATOM':
                    continue

                if line[5] in plddt.keys():
                    plddt[line[5]].append(float(line[10]))
                else:
                    plddt[line[5]] = [float(line[10])]

                if int(line[5]) in motif and line[2] in atoms:
                    pos = np.array([[line[6],line[7],line[8]]],dtype=float)
                    coord = np.concatenate((coord,pos),0) 

            for res in plddt.keys():
                plddts.append(np.mean(plddt[res]))                          
    # save_path = output_file
    return output_file,pdb_string, coord, mean_plddt, plddts, ptm, mean_pae


