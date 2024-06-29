import sys, os, argparse, copy, subprocess, glob, pickle, json, tempfile, random
# import time
from pathlib import Path
import numpy as np
from Bio import *
# import gc
import logging
import torch
import esm

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import inference_v1,loss,mutate
from inference_v1 import main
# t_init=time.time()
args = inference_v1.get_args()
atoms = (args.atom).split(',')
# load model
logging.info(f"Loading model")
# esm_model = esm.pretrained.esmfold_v1()
# esm_model = esm_model.eval().cuda()

def parse_fasta(fasta_string: str) :
  """Parses FASTA string and returns list of strings with amino-acid sequences.

  
  Arguments:
    fasta_string: The string contents of a FASTA file.     
    > with open(input_fasta_path) as f:
    >  input_fasta_str = f.read()

  Returns:
    A tuple of two lists:
    * A list of sequences.
    * A list of sequence descriptions taken from the comment lines. In the
      same order as the sequences.
  """
  sequences = []
  descriptions = []
  index = -1
  for line in fasta_string.splitlines():
    line = line.strip()
    if line.startswith('>'):
      index += 1
      descriptions.append(line[1:])  # Remove the '>' at the beginning.
      sequences.append('')
      continue
    elif not line:
      continue  # Skip blank lines.
    sequences[index] += line

  return sequences, descriptions

def enable_cpu_offloading(model):
    from torch.distributed.fsdp import CPUOffload, FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import enable_wrap, wrap

    torch.distributed.init_process_group(
        backend="nccl", init_method="tcp://localhost:9999", world_size=1, rank=0
    )

    wrapper_kwargs = dict(cpu_offload=CPUOffload(offload_params=True))

    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **wrapper_kwargs):
        for layer_name, layer in model.layers.named_children():
            wrapped_layer = wrap(layer)
            setattr(model.layers, layer_name, wrapped_layer)
        model = wrap(model)

    return model


def init_model_on_gpu_with_cpu_offloading(model):
    model = model.eval()
    model_esm = enable_cpu_offloading(model.esm)
    del model.esm
    model.cuda()
    model.esm = model_esm
    return model

esm_model = esm.pretrained.esmfold_v1()


esm_model = esm_model.eval()
esm_model.set_chunk_size(args.chunk_size)

if args.cpu_only:
    esm_model.esm.float()  
    esm_model.cpu()
elif args.cpu_offload:
    esm_model = init_model_on_gpu_with_cpu_offloading(esm_model)
else:
    esm_model.cuda()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

if not os.path.exists(args.final_des_dir):
    os.mkdir(args.final_des_dir)

logging.info(f'start hal')

AA_freq = {'A': 0.07421620506799341,
 'R': 0.05161448614128464,
 'N': 0.044645808512757915,
 'D': 0.05362600083855441,
 'C': 0.02468745716794485,
 'Q': 0.03425965059141602,
 'E': 0.0543119256845875,
 'G': 0.074146941452645,
 'H': 0.026212984805266227,
 'I': 0.06791736761895376,
 'L': 0.09890786849715096,
 'K': 0.05815568230307968,
 'M': 0.02499019757964311,
 'F': 0.04741845974228475,
 'P': 0.038538003320306206,
 'S': 0.05722902947649442,
 'T': 0.05089136455028703,
 'W': 0.013029956129972148,
 'Y': 0.03228151231375858,
 'V': 0.07291909820561925}
letters = {'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

mask_len = [int(i) for i in (args.mask_len).split(',')]
motif_id = (args.motif_id).split(',')
scaf_len = sum(mask_len)

from Bio.PDB import *
parser = PDBParser()
structure = parser.get_structure("ref", args.reference)
model = structure[0]
motif_seq={} #0:motif1_seq; 1:motif2_seq.
coord = []
dm_id = []
motif_start = mask_len[0]+1
for motif_idx,i in enumerate(motif_id):
    chain_id = i[0]
    s,e = [int(x) for x in i[1:].split('-')]

    motif_len = e-s+1
    dm_id+=(list(range(motif_start,motif_start+motif_len)))
    motif_start = motif_start+motif_len+mask_len[motif_idx+1]

    motif_seq[motif_idx] = ''
    chain = model[chain_id]
    residues = chain.get_residues()
    for res in residues:
        resname = res.get_resname()
        res_id = res.get_id()[1]
        if resname in letters.keys() and int(res_id) in range(s,e+1):
            motif_seq[motif_idx] += letters[resname]
            for atom in atoms:
                pos = res[atom]
                coord.append(pos.get_coord())
ref = np.array(coord)


if args.pre_sequence is None:
    num = 0
    sequences = []
    while num < args.number:
        mask_seq=[]
        for i in range(0,len(mask_len)):
            mask_seq.append(''.join(np.random.choice(list(AA_freq.keys()), size=mask_len[i], p=list(AA_freq.values()))))
        des_seq=''
        for i in range(0,len(motif_id)):
            des_seq+=mask_seq[i]+motif_seq[i]
        des_seq+=mask_seq[i+1]
        if des_seq not in sequences:
            logging.info(f"{des_seq}")
            sequences.append(des_seq)
            des_len = len(des_seq)
            num+=1
else:
    logging.info(f"Using pre-defined sequences {args.pre_sequence}")
    with open(args.pre_sequence, 'r') as f:
        fasta_string = f.read()
    sequences, de = parse_fasta(fasta_string)
    des_len = len(sequences[0])

des_seqs = []
fst_suc_step = []
for init_seq_idx, des_seq in enumerate(sequences):
    traj = []
    num = init_seq_idx
    # # write sequence into fas
    # seq0 = SeqRecord(Seq(des_seq),id="init_seq",description="")
    # my_records = [seq0]
    # output_fasta=f"{args.output_dir}/{num}_mut0.fas"
    # SeqIO.write(my_records, output_fasta, "fasta")

    desc = f'{num}_mut0'
    all_sequences = [(desc, des_seq)]

    # save_path,pdb,des_coord,plddt,plddts=main(esm_model,output_fasta,num,dm_id)
    save_path,pdb,des_coord,plddt,plddts, ptm, mean_pae=main(esm_model,all_sequences,num,dm_id)

    M = np.linspace(round(args.max_mut/100*scaf_len), 1, args.step) # stepped linear decay of the mutation rate

    for i in range(args.step):
        # Update a few things.
        T = args.t1*(np.exp(np.log(0.5) / args.t2) ** i) # update temperature
        n_mutation = round(M[i]) # update mutation rate
        accepted = False # reset

        if i == 0: # do a first pass through the network before mutating anything -- baseline
            rmsd,rot,tran=loss.get_rmsd(ref,des_coord)
            logging.info(f"RMSD is {rmsd}")
            logging.info(f"plddt is {plddt}")
            current_loss=100-plddt+args.loss*rmsd
            logging.info(f"current loss is {current_loss}")
            logging.info(f"{args.loss},{args.t1},{args.t2}")

            traj.append((i, desc, des_seq, pdb, rmsd, plddt, mean_pae, ptm, True))

        else:
            #introduce mutation
            sites=mutate.select_positions(plddts,n_mutation,dm_id, des_len, option='r')
            logging.info(f'mut{i} mutation sites: {sites}')
            mut_seq=mutate.random_mutate(des_seq,sites)
            logging.info(f'mut{i} seq: {mut_seq}')

            # seq = SeqRecord(Seq(mut_seq),id=f"mut{i}",description="")
            # records = [seq]
            # mut_fas=f"{args.output_dir}/{num}_mut{i}.fas"
            # SeqIO.write(records, mut_fas, "fasta")

            desc = f'{num}_mut{i}'
            all_sequences = [(desc, mut_seq)]

            #prediction
            save_path,pdb,mut_coord,mut_plddt,plddts, ptm, mean_pae=main(esm_model,all_sequences,num,dm_id)
            mut_rmsd,rot,tran=loss.get_rmsd(ref,mut_coord)
            logging.info(f"RMSD is {mut_rmsd}")
            try_loss=100-mut_plddt+args.loss*mut_rmsd

            delta = try_loss - current_loss
            logging.info(f'current loss is {current_loss}')
            logging.info(f'try loss is {try_loss}, {i}')

            # If the new solution is better, accept it.
            if delta < 0:
                accepted = True
                logging.info(f"do accept")
                current_loss = try_loss # accept loss change
                des_seq=mut_seq # accept the mutation
                rmsd=mut_rmsd
                plddt=mut_plddt

            # If the new solution is not better, accept it with a probability of e^(-cost/temp).
            else:

                if np.random.uniform(0, 1) < np.exp( -delta / T):
                    accepted = True
                    logging.info(f"do accept")
                    current_loss = try_loss # accept loss change
                    des_seq=mut_seq # accept the mutation
                    rmsd=mut_rmsd
                    plddt=mut_plddt

                else:
                    accepted = False
                    logging.info(f'not accept')

            traj.append((i, desc, des_seq, pdb, mut_rmsd, mut_plddt, mean_pae, ptm, accepted))
            
        if args.earlystop:
            if rmsd < 1 and plddt > 80:
                break

    des_seqs.append(des_seq)
    fst_suc_step.append(i)
    
    with open(f"{args.output_dir}/{args.bb_suffix}_{num}.pkl", 'wb') as f:
        pickle.dump(traj, f)

    # seq = SeqRecord(Seq(des_seq),id=f"final_des",description="")
    # records = [seq]
    # design_fas=f"{args.final_des_dir}/final_des{args.bb_suffix}_{num}.fas"
    # SeqIO.write(records, design_fas, "fasta")

    desc = f'final_des{args.bb_suffix}_{num}'
    all_sequences = [(desc, des_seq)]

    #预测
    save_path,pdb, des_coord,plddt,plddts, ptm, mean_pae=main(esm_model,all_sequences,f'final_des{args.bb_suffix}_{num}',dm_id)
    save_path = Path(f"{args.final_des_dir}/final_des{args.bb_suffix}_{num}.pdb")
    save_path.write_text(pdb)
    final_rmsd,rot,tran=loss.get_rmsd(ref,des_coord)
    logging.info(f'****** final_des{args.bb_suffix}_{num}: ,motif_RMSD:{final_rmsd}, plddt:{plddt} *******')
    t_end=time.time()

if len(des_seqs) != len(fst_suc_step):
    raise ValueError('different length of designed sequences with first success step list')
else:
    for i, seq in enumerate(fst_suc_step):
        with open(f"{args.output_dir}/early_stop_steps_{i}", 'wb') as f:
            f.write(f'{fst_suc_step[i]}')
        
    
# t_total = t_end-t_init
# logging.info(f'all time used {t_total} second')