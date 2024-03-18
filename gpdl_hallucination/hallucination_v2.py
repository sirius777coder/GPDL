import sys, os, argparse, copy, subprocess, glob, time, pickle, json, tempfile, random
import numpy as np
from Bio import *
import gc
import logging
import torch
import esm
import os

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO

import inference_v2,loss,mutate
from inference_v2 import main
t_init=time.time()
args = inference_v2.get_args()
atoms = (args.atom).split(',')
# load model
logging.info("Loading model")
# esm_model = esm.pretrained.esmfold_v1()
# esm_model = esm_model.eval().cuda()

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

print('start hal')

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


from Bio.PDB import *
parser = PDBParser()
structure = parser.get_structure("ref", args.reference)
model = structure[0]
motif_seq={} 
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
                # print(pos)
                coord.append(pos.get_coord())
                # print(coord)
            # ca = res["CA"]
            # coord.append(ca.get_coord())
ref = np.array(coord)

zn = np.array([14.480,0.071,15.316])
van_r = 1.252

num = 0
sequences = []
while num < args.number:
    mask_seq=[]
    for i in range(0,len(mask_len)):
        mask_seq.append(''.join(np.random.choice(list(AA_freq.keys()), size=mask_len[i], p=list(AA_freq.values()))))
    # from Bio.PDB import *
    des_seq=''
    for i in range(0,len(motif_id)):
        des_seq+=mask_seq[i]+motif_seq[i]
    des_seq+=mask_seq[i+1]
    if des_seq not in sequences:
        print(des_seq)
        sequences.append(des_seq)
        # print(motif_seq)
        des_len = len(des_seq)
        num+=1

des_seqs=[]
for i, des_seq in enumerate(sequences):
    num = i
    # 将序列写入fas文件
    seq0 = SeqRecord(Seq(des_seq),id="init_seq",description="")
    my_records = [seq0]
    output_fasta=f"{args.output_dir}/{num}_mut0.fas"
    # print(output_fasta)
    SeqIO.write(my_records, output_fasta, "fasta")

    save_path,pdb,des_coord,all_coord,plddt,plddts=main(esm_model,output_fasta,num,dm_id)

    M = np.linspace(args.n_mut_max, 1, args.step) # stepped linear decay of the mutation rate,step=5000
    for i in range(args.step):
        # Update a few things.
        T = args.t1*(np.exp(np.log(0.5) / args.t2) ** i) # update temperature
        n_mutation = round(M[i]) # update mutation rate
        accepted = False # reset

        if i == 0: # do a first pass through the network before mutating anything -- baseline
            rmsd,rot,tran=loss.get_rmsd(ref,des_coord)
            logging.info(f"RMSD is {rmsd}")
            print("plddt is", plddt)
            clash = loss.get_potential(zn, all_coord, rot, tran, van_r)

            current_loss=100-plddt+args.loss*rmsd+5*clash
            print('current loss is',current_loss)
            print('num of clash:',clash)
            print(args.loss,args.t1,args.t2)
        else:
            #introduce mutation
            sites=mutate.select_positions(plddts,n_mutation,dm_id, des_len, option='r')
            print(f'mut{i} mutation sites:',sites)
            mut_seq=mutate.random_mutate(des_seq,sites)
            print(f'mut{i} seq:',mut_seq)
            seq = SeqRecord(Seq(mut_seq),id=f"mut{i}",description="")
            records = [seq]
            mut_fas=f"{args.output_dir}/{num}_mut{i}.fas"
            SeqIO.write(records, mut_fas, "fasta")

            #prediction
            save_path,pdb,mut_coord,all_coord,mut_plddt,plddts=main(esm_model,mut_fas,num,dm_id)
            mut_rmsd,rot,tran=loss.get_rmsd(ref,mut_coord)
            logging.info(f"RMSD is {mut_rmsd}")
            clash = loss.get_potential(zn, all_coord, rot, tran, van_r)
            try_loss=100-mut_plddt+args.loss*mut_rmsd+5*clash
            print('num of clash:',clash)
            
            delta = try_loss - current_loss # all losses must be defined such that optimising equates to minimising.
            print('current loss is',current_loss)
            print('try loss is',try_loss,i)


            # If the new solution is better, accept it.
            if delta < 0:
                accepted = True
                print("do accept")
                current_loss = try_loss # accept loss change
                des_seq=mut_seq # accept the mutation
                rmsd=mut_rmsd
                plddt=mut_plddt

            # If the new solution is not better, accept it with a probability of e^(-cost/temp).
            else:

                if np.random.uniform(0, 1) < np.exp( -delta / T):
                    accepted = True
                    print("do accept")
                    current_loss = try_loss # accept loss change
                    des_seq=mut_seq # accept the mutation

                else:
                    accepted = False
                    print('not accept')

    des_seqs.append(des_seq)

for num,des_seq in enumerate(des_seqs):
    seq = SeqRecord(Seq(des_seq),id=f"final_des",description="")
    records = [seq]
    design_fas=f"{args.final_des_dir}/final_des{args.bb_suffix}_{num}.fas"
    SeqIO.write(records, design_fas, "fasta")

    #预测
    save_path,pdb, des_coord,all_coord,plddt,plddts=main(esm_model,design_fas,f'final_des{num}',dm_id)
    save_path.write_text(pdb)
    final_rmsd,rot,tran=loss.get_rmsd(ref,des_coord)
    clash = loss.get_potential(zn, all_coord, rot, tran, van_r)
    try_loss=100-plddt+args.loss*final_rmsd+5*clash
    print(f'****** final_des{num}: ',f'loss:{try_loss}',f'motif_RMSD:{final_rmsd}',f'plddt:{plddt}',f'clash:{clash} *******')
    t_end=time.time()
    print('all time used',t_end-t_init,"second")