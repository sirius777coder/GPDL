import io
import os
import sys
import time
from typing import Sequence

import argparse
import pdbfixer
import openmm
from openmm import unit
from openmm import app as openmm_app
from openmm.app.internal.pdbstructure import PdbStructure

# user-defined parameter
parser = argparse.ArgumentParser()
parser.add_argument('--inpath', '-i', type=str, default='./pdbs/',
                    help='input path')
parser.add_argument('--outpath', '-o', type=str, default='./refined/',
                    help='output path')
args = parser.parse_args()

# default parameter settings
max_iterations = 0
tolerance = 2.39
stiffness = 10.0
exclude_residues = []
max_attempts = 1000
use_gpu = False
ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms


def fix_structure(pdbfile):
    """Fix the input PDB file with pdbfixer"""
    fixer = pdbfixer.PDBFixer(pdbfile=pdbfile)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    fixer.addMissingHydrogens()

    out_handle = io.StringIO()
    openmm_app.PDBFile.writeFile(
        fixer.topology, fixer.positions, out_handle, keepIds=True
    )
    return out_handle.getvalue()


def _add_restraints(
    system: openmm.System,
    reference_pdb: openmm_app.PDBFile,
    stiffness: unit.Unit,
    rset: str,
    exclude_residues: Sequence[int],
):
    """Adds a harmonic potential that restrains the system to a structure."""
    assert rset in ["non_hydrogen", "c_alpha"]

    force = openmm.CustomExternalForce(
        "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    force.addGlobalParameter("k", stiffness)
    for p in ["x0", "y0", "z0"]:
        force.addPerParticleParameter(p)

    for i, atom in enumerate(reference_pdb.topology.atoms()):
        if atom.residue.index in exclude_residues:
            continue
        if will_restrain(atom, rset):
            force.addParticle(i, reference_pdb.positions[i])
    system.addForce(force)


def will_restrain(atom: openmm_app.Atom, rset: str) -> bool:
    """Returns True if the atom will be restrained by the given restraint set."""

    if rset == "non_hydrogen":
        return atom.element.name != "hydrogen"
    elif rset == "c_alpha":
        return atom.name == "CA"


def minimize(
    pdb_str: str,
    max_iterations: int,
    tolerance: unit.Unit,
    stiffness: unit.Unit,
    restraint_set: str,
    exclude_residues: Sequence[int],
    use_gpu: bool,
):
    """Minimize energy via openmm"""
    pdb_file = io.StringIO(pdb_str)
    pdb = openmm_app.PDBFile(pdb_file)

    force_field = openmm_app.ForceField("amber99sb.xml")
    constraints = openmm_app.HBonds
    system = force_field.createSystem(pdb.topology, constraints=constraints)
    
    if stiffness > 0 * ENERGY / (LENGTH ** 2):
        _add_restraints(system, pdb, stiffness, restraint_set, exclude_residues)

    integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
    platform = openmm.Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
    simulation = openmm_app.Simulation(
        pdb.topology, system, integrator, platform
    )
    simulation.context.setPositions(pdb.positions)

    ret = {}
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    simulation.minimizeEnergy(maxIterations=max_iterations, tolerance=tolerance)
    state = simulation.context.getState(getEnergy=True, getPositions=True)
    ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
    ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
    ret["min_pdb"] = _get_pdb_string(simulation.topology, state.getPositions())
    return ret


def _get_pdb_string(topology: openmm_app.Topology, positions: unit.Quantity):
    """Returns a pdb string provided OpenMM topology and positions."""
    with io.StringIO() as f:
        openmm_app.PDBFile.writeFile(topology, positions, f)
        return f.getvalue()


tolerance = tolerance * ENERGY
stiffness = stiffness * ENERGY / (LENGTH ** 2)

if not os.path.exists(args.inpath):
    os.makedirs(args.inpath)
if not os.path.exists(args.outpath):
    os.makedirs(args.outpath)

start = time.perf_counter()

for pdbs in os.listdir(args.inpath):
    if pdbs.endswith('.pdb'):
        pdb_file = open(os.path.join(args.inpath, pdbs), 'r')
        pdb_string = fix_structure(pdb_file)

        minimized = False
        attempts = 0
        fails = []
        while not minimized and attempts < max_attempts:
            attempts += 1
            try:
                ret = minimize(
                    pdb_string,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                    stiffness=stiffness,
                    restraint_set="non_hydrogen",
                    exclude_residues=exclude_residues,
                    use_gpu=use_gpu,
                )
                minimized = True
            except Exception as e:  # pylint: disable=broad-except
                print(e)
        if not minimized:
            fails.append(pdbs)
            # raise ValueError(f"Minimization failed after {max_attempts} attempts.")
        ret["min_attempts"] = attempts

        with open(os.path.join(args.outpath, pdbs), 'w') as f:
            f.write(ret["min_pdb"])

print("optimization using %.2f s" % (time.perf_counter() - start))
if len(fails) != 0:
    print("fail to optimize %d structures, filename list:\n%s" % tuple(fails.insert(0, len(fails))))
