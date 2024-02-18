import biotite.structure
from biotite.structure.io import pdbx, pdb
from biotite.structure.residues import get_residues
from biotite.structure import filter_backbone
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence
import biotite.structure.io as strucio
import biotite.structure as struc
import sys


restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}
restype_num = len(restypes)  # := 20.
alphabet = {
    'ALA': 'A', 'VAL': 'V', 'PHE': 'F', 'PRO': 'P', 'MET': 'M',
    'ILE': 'I', 'LEU': 'L', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'TYR': 'Y', 'HIS': 'H',
    'CYS': 'C', 'ASN': 'N', 'GLN': 'Q', 'TRP': 'W', 'GLY': 'G',
}
alphabet_map = {v:k for k,v in alphabet.items()}


def extract_seq(protein, chain_id=None):
    if isinstance(protein, str):
        atom_array = strucio.load_structure(protein, model=1)
    elif isinstance(protein, struc.AtomArrayStack):
        atom_array = protein[0]
    elif isinstance(protein, struc.AtomArray):
        atom_array = protein
    aa_mask = struc.filter_canonical_amino_acids(atom_array)
    atom_array = atom_array[aa_mask]
    all_chains = get_chains(atom_array)
    if len(all_chains) == 0:
        raise ValueError('No chains found in the input file.')
    if chain_id is None:
        chain_ids = all_chains
    elif isinstance(chain, list):
        chain_ids = chain
    else:
        chain_ids = [chain]
    for chain in chain_ids:
        if chain not in all_chains:
            raise ValueError(f'Chain {chain} not found in input file')
    chain_filter = [a.chain_id in chain_ids for a in atom_array]
    atom_array = atom_array[chain_filter]
    # mask canonical aa
    aa_mask = struc.filter_amino_acids(atom_array)
    atom_array = atom_array[aa_mask]
    residue_identities = get_residues(atom_array)[1]
    seq = ''.join([ProteinSequence.convert_letter_3to1(r)
                  for r in residue_identities])
    return seq

if __name__ == "__main__":
    print(f"{extract_seq(sys.argv[1])}")