import argparse
import os

'''
This is a script adding chains in PDB files generated by MD.
For some tools like ProteinMPNN, the chain ID is needed, which is erased when running MD analysis.
'''

def add_chain_id(pdb_file, chain_id):
    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if line.startswith('ATOM'):
            line = line[:21] + chain_id + line[22:]
        new_lines.append(line)

    new_pdb_file = pdb_file.split('.pdb')[0] + '_new.pdb'
    with open(new_pdb_file, 'w') as file:
        file.writelines(new_lines)

    print(f"Chain ID '{chain_id}' added to the PDB file '{pdb_file}'. New file saved as '{new_pdb_file}'.")


def run_script(directory, chain_id):
    for file_name in os.listdir(directory):
        if file_name.endswith(".pdb"):
            pdb_file = os.path.join(directory, file_name)
            add_chain_id(pdb_file, chain_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Add chain ID to PDB files.')
    parser.add_argument('--dir', default="./",help='Directory containing PDB files')
    parser.add_argument('--chain_id', default='A', help='Chain ID to be added (default: A)')

    args = parser.parse_args()

    if args.dir:
        run_script(args.dir, args.chain_id)
    else:
        print("Please provide a directory using the '--dir' argument.")

