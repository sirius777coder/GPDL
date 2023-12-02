import biotite.structure as struc
import biotite.structure.io as strucio
import os
import numpy as np
import json
import utils


def change_case(s, indices,mode=None):
    modified_string = []
    for i, char in enumerate(s):
        if i in indices:
            if mode == "upper":
                modified_string.append(char.upper())
            elif mode == "lower":
                modified_string.append(char.lower())
            elif mode == "exchange":
                modified_string.append(char.swapcase())
        else:
            modified_string.append(char)
    return ''.join(modified_string)

# 输入都是一段文字A119-140,A63-82,分段解析这个motif位置
def motif_extract(position,array,atom_part="all"):
    """
    subtract chain_id,res_id & atom option
    ignore the hetero elements

    DEMO:
    position = "A119-140,A63-82"
    motif_AtomArray = motif_extract(position,array,atom_part="backbone")
    """
    position = position.split(",")
    motif_array = []
    for i in position:
        chain_id = i[0]
        i = i.replace(chain_id,"")
        if "-" not in i: # single motif positions
            start = end = int(i)
        else:
            start,end = i.split("-")
            start,end = int(start),int(end)
        if atom_part == "all":
            motif_array.append(array[(array.chain_id==chain_id) & (array.res_id <= end) & (array.res_id >= start) & (array.hetero==False)])
        elif atom_part == "ca":
            motif_array.append(array[(array.chain_id==chain_id) & (array.res_id <= end) & (array.res_id >= start) & (array.hetero==False) & (array.atom_name=="CA")])
        elif atom_part == "backbone":
            motif_array.append(array[(array.chain_id==chain_id) & (array.res_id <= end) & (array.res_id >= start) & (array.hetero==False) & ((array.atom_name=="N") | (array.atom_name=="CA")| (array.atom_name=="C") | (array.atom_name=="O"))])
    motif_AtomArray = motif_array[0]
    for i in range(len(motif_array)-1):
        motif_AtomArray += motif_array[i+1]
    return motif_AtomArray

# original : [{'mask': 9}, {'A': [119, 140]}, {'mask': 18}, {'A': [63, 82]}, {'mask': 28}]
# convert to 
def convert_motif(des_seq):
    # des = [{'mask': 9}, {'A': [119, 140]}, {'mask': 18}, {'A': [63, 82]}, {'mask': 28}]
    length = 0
    motif_des_positon = ""
    for i in des_seq:
        if "mask" in i.keys():
            length += int(i['mask'])
        else:
            chain = list(i.keys())[0] # only one chain in an element
            if len(i[chain]) == 1:
                start = end = i[chain]
            else:
                start,end = i[chain]
            if motif_des_positon == "":
                motif_des_positon += f"A{length+1}-{length+end-start+1}"
            else:
                motif_des_positon += f",A{length+1}-{length+end-start+1}"
            length += end-start+1
    return motif_des_positon


def compute_rmsd(name,motif_native_pos,option=None,num_des=1000,atom_part="ca"):
    # # 0-30,A119-140,15-40,A63-82, 0-30
    reference = f"./benchmark_set/{name.upper()}.pdb"
    array = strucio.load_structure(reference,model=1)

    # motif_native_pos = "A1-7,A28-79"
    motif_AtomArray = motif_extract(motif_native_pos,array,atom_part=atom_part)

    if option is None:
        with open(f"./design/{name.lower()}/{name.upper()}.txt","r") as f:
            info = [i.replace("\n","") for i in f.readlines()]
    else:
        str1 = change_case(option,indices=[_ for _ in range(4)],mode="lower")
        str2 = change_case(option,indices=[_ for _ in range(4)],mode="upper")
        with open(f"./design/{str1}/{str2}.txt","r") as f:
            info = [i.replace("\n","") for i in f.readlines()]
    count = 0 
    motif_region = {"T1":{},"T2":{},"T3":{},"T1_max_2":{}}
    for i in range(1,len(info),2):
        count += 1
        if count <= num_des:
            motif_region["T1"][count-1] = info[i]
        else:
            break

    rmsd = {"T1":[],"T2":[],"T1_max":[],"T1_max_2":[]}
    plddt = []
    min_rmsd = 100
    idx = 0
    for stage in ["T1"]:
        num = 0
        for pdb in range(0,num_des):
            motif_region_des = convert_motif(json.loads(motif_region[stage][pdb].replace("'","\"")))
            if option is None:
                file = f"./design/{name.lower()}/{name.upper()}_{pdb}.pdb"
            else:
                file = f"./design/{str1}/{str2}_{pdb}.pdb"
            if not os.path.exists(file):
                continue
            des = strucio.load_structure(file)
            des_AtomArray = motif_extract(motif_region_des,des,atom_part=atom_part)
            if len(des_AtomArray) != len(motif_AtomArray):
                continue
            superimposed, _ = struc.superimpose(motif_AtomArray, des_AtomArray)
            rms = struc.rmsd(motif_AtomArray, superimposed)
            if rms < min_rmsd:
                min_rmsd = rms
                idx = pdb
            rmsd[stage].append(rms)
            plddt_single,mean_plddt = utils.extract_plddt(file)
            plddt.append(mean_plddt)
    n=200
    rms = np.array(rmsd['T1'])
    rms_idx = np.argsort(rms)[:n]
    min_n = rms[rms_idx]

    if option is not None:
        print(option)
    else:
        print(name)
    for i in range(len(min_n)):
        if min_n[i] < 2.0:
            print(f"{rms_idx[i]},{min_n[i]},{plddt[rms_idx[i]]}")


if __name__ == "__main__":
    # print("5trv_short")
    # compute_rmsd("5trv", "A45-65", "5trv_short",
    #              num_des=36900, atom_part="backbone")
    # print("5trv_med")
    # compute_rmsd("5trv", "A45-65", "5trv_med",
    #              num_des=33000, atom_part="backbone")

    # compute_rmsd("5trv","A45-65","5trv_long",num_des=27500,atom_part="backbone")

    # compute_rmsd("6exz", "A558-572", "6exz_short",
    #              num_des=1000, atom_part="backbone")

    # compute_rmsd("6exz", "A558-572", "6exz_med",
    #              num_des=1000, atom_part="backbone")

    # compute_rmsd("6exz", "A558-572", "6exz_long",
    #              num_des=1000, atom_part="backbone")

    # compute_rmsd("2KL8", "A1-7,A28-79", num_des=1000, atom_part="backbone")

    # compute_rmsd("1BCF", "A92-99,A123-130,A47-54,A18-25",num_des=27974, atom_part="backbone")
    
    # print("5IUS backbone-----")
    # compute_rmsd("5IUS","A119-140,A63-82",num_des=10000, atom_part="ca")
    print("5IUS backbone")
    compute_rmsd("5IUS","A119-140,A63-82",num_des=32000, atom_part="backbone")

    # print("5yui backbone")
    # compute_rmsd("5YUI","A93-97,A118-120,A198-200",num_des=34500, atom_part="backbone")
    # print("5yui ca")
    # compute_rmsd("5YUI","A93-97,A118-120,A198-200",num_des=34500, atom_part="ca")
    # compute_rmsd("6VW1", "A24-42,A64-82", num_des=31340, atom_part="backbone")

    # compute_rmsd("1QJG", "A38,A14,A99", num_des=11200, atom_part="backbone")

    # compute_rmsd("3IXT", "P254-277", num_des=1000, atom_part="backbone")

    # compute_rmsd("1YCR", "B19-27", num_des=1000, atom_part="backbone")

    # compute_rmsd("7MRX", "B25-46", "7MRX_60",
    #              num_des=1000, atom_part="backbone")

    # compute_rmsd("7MRX", "B25-46", "7mrx_85",
    #              num_des=27000, atom_part="backbone")

    # compute_rmsd("7MRX", "B25-46", "7mrx_128",
    #              num_des=19855, atom_part="backbone")

    # compute_rmsd("4JHW","F196-212,F63-69",num_des=30000, atom_part="backbone")

    # compute_rmsd("4ZYP", "A422-436", num_des=26000, atom_part="backbone")

    # compute_rmsd("5WN9", "A170-189", num_des=35615, atom_part="backbone")

    # compute_rmsd("1prw", "A16-35,A52-71", num_des=1000, atom_part="backbone")

    # compute_rmsd("1ycr", "B19-27", num_des=1000, atom_part="backbone")

    # compute_rmsd("5tpn", "A163-181", num_des=34755, atom_part="backbone")

    # compute_rmsd("2kl8", "A1-7,A28-79", num_des=1000, atom_part="backbone")

    # compute_rmsd("6e6r", "A23-35", "6e6r_short",
    #              num_des=1000, atom_part="backbone")

    # compute_rmsd("6e6r", "A23-35", "6e6r_med",
    #              num_des=1000, atom_part="backbone")

    # compute_rmsd("6e6r", "A23-35", "6e6r_long",
    #              num_des=1000, atom_part="backbone")