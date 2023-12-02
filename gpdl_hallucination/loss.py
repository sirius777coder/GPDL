from turtle import shape
from Bio.PDB import *
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
import sys, os, argparse, copy, subprocess, glob, time, pickle, json, tempfile, random

def get_coord (pdb, m_id, chain, option="CA"):
    paser = PDBParser()
    structure = paser.get_structure("pdb", pdb)

    res_dict={}#ref蛋白所有AA
    for residue in structure.get_residues():
        res_idx=int(str(residue).split()[3].split("=")[1]) #得到所有残基的序号
        res_dict[res_idx]=residue #字典【氨基酸序号】=氨基酸(生成器)
        # print(res_idx)

    if option == "CA":
        coord = []
        # x=0
        model = structure[0]
        chain = model[chain]
        for id in m_id:
            res = res_dict[id]
            ca = res["CA"]
            coord.append(ca.get_coord())
        coord = np.array(coord)
        return coord

def get_rmsd(ref_coord,des_coord):
    sup = SVDSuperimposer()
    sup.set(ref_coord, des_coord)
    sup.run()
    motif_rmsd = sup.get_rms()
    rot, tran = sup.get_rotran()
    return motif_rmsd,rot,tran

def get_lddt(pdb):
    plddt = {} #res_id:[plddt],omfold res_id start from 0
    plddts = [] #every res
    lddt=[] #every atom --- average
    with open(pdb) as f:
        atom = 0 
        for line in f.readlines():
            line = line.replace("\n", "").split()
            if line[0] != 'ATOM':
                continue
            else:
                if line[5] in plddt.keys():
                    plddt[line[5]].append(float(line[10]))
                else:
                    plddt[line[5]] = [float(line[10])]
                    
    for res in plddt.keys():
        plddts.append(np.mean(plddt[res]))
        lddt+=plddt[res]
    lddt=np.mean(lddt)
    return plddt,plddts,lddt

def get_potential(po, all_coord, rot, tran, van_r):
    clash = 0
    coord = np.dot(all_coord, rot) + tran
    for i in coord:
        distance = np.linalg.norm(i-po)
        if distance < van_r: #1.252
            clash += 1
    return clash
            
