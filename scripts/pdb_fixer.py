from pdbfixer import PDBFixer
from openmm.app import PDBFile
import subprocess
import mdtraj as md
import numpy as np
import time
import sys
import argparse
def pdb_fix(traj_name,temp_fix_dir = '/path/to/EGDiff/data/temp_fix'):
    traj = md.load(traj_name)
    for i, frame in enumerate(traj):
        frame.save_pdb('{}/temp_{}.pdb'.format(temp_fix_dir,i))
    for i in range(len(traj)):
        # Load each frame
        fixer = PDBFixer(filename='{}/temp_{}.pdb'.format(temp_fix_dir,i))

        # Find and add missing sidechains
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens()

        # Save the fixed frame
        with open('{}/temp_{}.pdb'.format(temp_fix_dir,i), 'w') as output:
            PDBFile.writeFile(fixer.topology, fixer.positions, output)
    return traj.xyz.shape[0]



def create_openmm_script(pdb_file, gpu_index, base_directory="/path/to/EGDiff/data/temp_fix"):
    return f"""
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
import mdtraj as md
# Load the PDB file
pdb = PDBFile('{base_directory}/{pdb_file}')
forcefield = ForceField('amber99sb.xml', 'tip3p.xml')

# Create the modeller to add solvent
modeller = Modeller(pdb.topology, pdb.positions)
modeller.addSolvent(forcefield, model='tip3p', padding=1.0*nanometers)

# Create the system
system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1*nanometer, constraints=HBonds)

# Create the integrator
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds)

# Create the simulation
platform = Platform.getPlatformByName('CUDA')
properties = {{'DeviceIndex': '{gpu_index}', 'CudaPrecision': 'mixed'}}
simulation = Simulation(modeller.topology, system, integrator, platform, properties)

# Set the initial positions
simulation.context.setPositions(modeller.positions)

# Minimize the energy
simulation.minimizeEnergy()

# Equilibrate
simulation.context.setVelocitiesToTemperature(300*kelvin)
simulation.step(100)

# Get the final state
state = simulation.context.getState(getPositions=True)
final_positions = state.getPositions()

# Convert final positions to MDTraj trajectory
top = md.Topology.from_openmm(modeller.topology)
traj = md.Trajectory(final_positions.value_in_unit(nanometers), top)

# Select only protein atoms
protein_atoms = traj.topology.select('protein')
protein_traj = traj.atom_slice(protein_atoms)
protein_traj.save_pdb('{base_directory}/{pdb_file}')
"""


# Save the scripts to files
def write_script(frame_num,gpu_list = [0,1,2,3],temp_fix_dir = '/path/to/EGDiff/data/temp_fix'):
    for i in range(frame_num):
        gpu_index = gpu_list[i//len(gpu_list)]
        with open('{}/temp_{}.py'.format(temp_fix_dir, i), 'w') as f:
            f.write(create_openmm_script('temp_{}.pdb'.format(i), gpu_index))


def run_subprocess(frame_num):
    # Run the simulations
    processes = []
    for i in range(frame_num):
        processes.append(subprocess.Popen(['python', '/homes/liu3307/str2str_analysis/fix/temp_{}.py'.format(i)]))

    # Wait for all processes to finish
    for p in processes:
        p.wait()


def merge(original_file_name,frame_num, base_directory="/path/to/EGDiff/data/temp_fix"):
    # Initialize a list to store the trajectories with only protein atoms
    protein_trajectories = []

    # Loop over each (xtc, pdb) pair
    count = 0
    traj_true = 0
    for i in range(frame_num):
        # Load the trajectory and corresponding pdb file
        try:
            traj = md.load('{}/temp_{}.pdb'.format(base_directory, i))

            # Select only the protein atoms
            protein_atoms = traj.topology.select('protein and (name N or name CA or name C or name O)')
            # protein_atoms = traj.topology.select('protein')
            protein_traj = traj.atom_slice(protein_atoms)
            if i == 0:
                traj_true = protein_traj.xyz.shape[1]
            if protein_traj.xyz.shape[1] == traj_true:
                protein_trajectories.append(protein_traj)
            else:
                print("{}: shape error".format(i))
            count += 1
            # print("{}: time {}".format(count,time.time() - start_time))
            count += 1
        except Exception as e:
            print("{}: fail".format(count))
            count += 1
            continue
    # Concatenate all the protein trajectories
    merged_protein_trajectory = md.join(protein_trajectories)
    base_name, extension = original_file_name.split('.')
    new_A = f"{base_name}_fixed.{extension}"
    merged_protein_trajectory.save_pdb(new_A)

def pdb_fix_all(original_file_name,gpu_list = [0],temp_fix_dir = '/path/to/EGDiff/data/temp_fix'):
    num_frame = pdb_fix(original_file_name)
    write_script(frame_num=num_frame,gpu_list  = gpu_list,temp_fix_dir=temp_fix_dir)
    run_subprocess(num_frame)
    merge(original_file_name,num_frame)


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run pdb_fix_all with optional inputs.")

    # Required argument: original_file_name
    parser.add_argument('original_file_name', type=str, help="Path to the original PDB file")

    # Optional argument: gpu_list, default is [0]
    parser.add_argument('--gpu_list', type=str, default="0",
                        help="Comma-separated list of GPU indices (default is [0])")

    # Optional argument: temp_fix_dir, default is '/path/to/EGDiff/data/temp_fix'
    parser.add_argument('--temp_fix_dir', type=str, default='/path/to/EGDiff/data/temp_fix',
                        help="Temporary fix directory (default is '/path/to/EGDiff/data/temp_fix')")

    # Parse the arguments
    args = parser.parse_args()
    # Convert the gpu_list from a string to a list of integers
    gpu_list = [int(gpu) for gpu in args.gpu_list.split(',')]

    # Call the pdb_fix_all function with the provided or default arguments
    pdb_fix_all(args.original_file_name, gpu_list=gpu_list, temp_fix_dir=args.temp_fix_dir)
