import biotite
import biotite.structure as struc
import biotite.structure.io as strucio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import scipy.stats as sts
import os

# for the short 5trv_short
# save_dihedral = {}
# dir_5trv_short = "/lustre/home/acct-stu/stu005/ESM-Inpainting/esm_inpaint/design/5trv_short/"
# phi_short = np.array([])
# psi_short = np.array([])
# omega_short = np.array([])
# for pdb_name in os.listdir(dir_5trv_short):
#     file = os.path.join(dir_5trv_short,pdb_name)
#     atom_array = strucio.load_structure(file)
#     phi, psi, omega = struc.dihedral_backbone(
#         atom_array[atom_array.chain_id == "A"]
#     )
#     # Conversion from radians into degree
#     phi *= 180/np.pi
#     psi *= 180/np.pi
#     omega *= 180/np.pi
#     # Remove invalid values (NaN) at first and last position
#     phi= phi[1:-1]
#     psi= psi[1:-1]
#     omega = omega[1:-1]
#     phi_short = np.append(phi_short, phi)
#     psi_short = np.append(psi_short, psi)
#     omega_short = np.append(omega_short, omega)
#     if len(phi_short) > 100000:
#         break
# # save phi and psi to a dic
# save_dihedral["short"] = {"phi":phi_short,"psi":psi_short,"omega":omega_short}

# dir_5trv_med = "/lustre/home/acct-stu/stu005/ESM-Inpainting/esm_inpaint/design/5trv_med/"
# phi_med = np.array([])
# psi_med = np.array([])
# omega_med = np.array([])
# for pdb_name in os.listdir(dir_5trv_med):
#     file = os.path.join(dir_5trv_med,pdb_name)
#     atom_array = strucio.load_structure(file)
#     phi, psi, omega = struc.dihedral_backbone(
#         atom_array[atom_array.chain_id == "A"]
#     )
#     # Conversion from radians into degree
#     phi *= 180/np.pi
#     psi *= 180/np.pi
#     omega *= 180/np.pi
#     # Remove invalid values (NaN) at first and last position
#     phi= phi[1:-1]
#     psi= psi[1:-1]
#     omega = omega[1:-1]
#     phi_med = np.append(phi_med, phi)
#     psi_med = np.append(psi_med, psi)
#     omega_med = np.append(omega_med, omega)
#     if len(phi_med) > 100000:
#         break
# save_dihedral["med"] = {"phi":phi_med,"psi":psi_med,"omega":omega_med}

# dir_5trv_long = "/lustre/home/acct-stu/stu005/ESM-Inpainting/esm_inpaint/design/5trv_long/"
# phi_long = np.array([])
# psi_long = np.array([])
# omega_long = np.array([])
# for pdb_name in os.listdir(dir_5trv_long):
#     file = os.path.join(dir_5trv_long,pdb_name)
#     atom_array = strucio.load_structure(file)
#     phi, psi, omega = struc.dihedral_backbone(
#         atom_array[atom_array.chain_id == "A"]
#     )
#     # Conversion from radians into degree
#     phi *= 180/np.pi
#     psi *= 180/np.pi
#     omega *= 180/np.pi
#     # Remove invalid values (NaN) at first and last position
#     phi= phi[1:-1]
#     psi= psi[1:-1]
#     omega = omega[1:-1]
#     phi_long = np.append(phi_long, phi)
#     psi_long = np.append(psi_long, psi)
#     omega_long = np.append(omega_long, omega)
#     if len(phi_long) > 100000:
#         break
# save_dihedral["long"] = {"phi":phi_long,"psi":psi_long,"omega":omega_long}

save_dihedral = {}
dir_2kl8 = "/lustre/home/acct-stu/stu005/ESM-Inpainting/esm_inpaint/backup/5ius"
phi_long = np.array([])
psi_long = np.array([])
omega_long = np.array([])
for pdb_name in os.listdir(dir_2kl8):
    file = os.path.join(dir_2kl8,pdb_name)
    atom_array = strucio.load_structure(file)
    phi, psi, omega = struc.dihedral_backbone(
        atom_array[atom_array.chain_id == "A"]
    )
    # Conversion from radians into degree
    phi *= 180/np.pi
    psi *= 180/np.pi
    omega *= 180/np.pi
    # Remove invalid values (NaN) at first and last position
    phi= phi[1:-1]
    psi= psi[1:-1]
    omega = omega[1:-1]
    phi_long = np.append(phi_long, phi)
    psi_long = np.append(psi_long, psi)
    omega_long = np.append(omega_long, omega)
    if len(phi_long) > 1000000:
        break
save_dihedral["5ius"] = {"phi":phi_long,"psi":psi_long,"omega":omega_long}

# save the dic to the numpy file
np.save("./dihedral_5ius.npy",save_dihedral,allow_pickle=True)