## This is a script that calculates the pairwise calpha contact probability between two proteins in a dimeric ensemble ##
## example command: python inter_pairwise_contact_prob_calc.py top.pdb traj.xtc weights.npy 1 393 395 787 inter_cmap.npy
# import modules
import sys
import numpy as np
import MDAnalysis as md
import mdtraj as mdj
import itertools

# set variables
path2top = sys.argv[1] # path to topology file of ensemble (.pdb or .gro)
path2traj = sys.argv[2] # path to ensemble trajectory file (.xtc)
path2weights = sys.argv[3] # path to weights from metadynamics or reweighting (.npy)
firstres_mol1 = sys.argv[4] # residue ID (1-indexed) for the first residue in one protein (nascent AAT: residues 1–393) included in the square matrix
lastres_mol1 = sys.argv[5] # residue ID (1-indexed) for the last residue included in one protein (nascent AAT: residues 1–393) in the square matrix
firstres_mol2 = sys.argv[6] # residue ID (1-indexed) for the first residue in the other protein (post AAT: residues 395–787) included in the square matrix
lastres_mol2 = sys.argv[7] # residue ID (1-indexed) for the last residue included in the other protein (post AAT: residues 395–787) in the square matrix
cutoff = 10.0 # cutoff distance for calpha contacts (10 angstroms)
path2savedcmap = sys.argv[8]

# Set universe for contact calculation
u = md.Universe(str(path2top),str(path2traj))

mol1_resi = u.select_atoms(f'resid {firstres_mol1}:{lastres_mol1} and name CA').resids
mol2_resi  = u.select_atoms(f'resid {firstres_mol2}:{lastres_mol2} and name CA').resids

# Build inter-chain contact index pairs for MDTraj (0-based index)
# nc:      residues 1–393     → MDTraj indices 0–392
# r chain: residues 395–787   → MDTraj indices 394–786
contact_indices = np.array(list(itertools.product(range(firstres_mol1-1, lastres_mol1-1),
                                                  range(firstres_mol2-1, lastres_mol2-1))))

# Ca-Ca distances over the ensemble
# THIS WILL CALCULATE A SQUARE MATRIX WITH INTER-RES1UE CONTACT PROABILITIES
# SPECIFIED: CA-CA DISTANCES USED, CONTACT-CUTOFF = 10.0A
print("Load subensemble trajectory - mdtraj ...", flush=True)
model = mdj.load(path2traj,top=path2traj)

# MD ensemble --> distance matrix
print("Calculating inter-residue distances for each frame...", flush=True)
contact_indices = np.array(list(itertools.combinations(range(nresi), 2)))
distances = mdj.compute_contacts(model, contacts = contact_indices, scheme = 'ca',periodic = False)
n_frames = distances.shape[0]

# Convert to Å (MDTraj returns nm)
distances_A = distances * 10.0

# Reshape directly into (n_frames, nres1, nres2) e.g. (n_frames, 393, 393)
# The order in contact_indices matches row-major fill of this reshape!
dist_matrix = distances_A.reshape(n_frames, len(mol1_resi), len(mol2_resi)

# Contact cutoff 10 Å → 1 = contact, 0 = no contact
prob_matrix = (dist_matrix <= cutoff).astype(np.float32)

# Average over frames → final 393×393 inter-chain probability map
wopt = np.load(str(path2weights))
wopt = wopt.ravel() # reshape
prob_avg = np.avg(prob_matrix,weights=wopt,axis=0)
np.save(str(path2savedcmap),prob_avg)
print("Pairwise Contact Probility Map Done!", flush=True)

