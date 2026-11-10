## This is a script that calculates the pairwise calpha contact probability for an ensensemble ##
## It also calculates the normalised sum contact probability to provide a per residue contact probability involvement ##
## Example command: python pairwise_contact_prob_calc.py top.pdb traj.xtc weights.npy 1 393 cmap.npy norm_cprob.npy

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
firstres = sys.argv[4] # residue ID (1-indexed) for the first residue included in the square matrix
lastres = sys.argv[5] # residue ID (1-indexed) for the last residue included in the square matrix
cutoff = 10.0 # cutoff distance for calpha contacts (10 angstroms)
path2savedcmap = sys.argv[6]
path2savednormcprob = sys.argv[7]

# Set universe for contact calculation
u = md.Universe(str(path2top),str(path2traj))
ca = u.select_atoms(f'resid {firstres}:{lastres} and name CA')
nresi = u.select_atoms(f'resid {firstres}:{lastres} and name CA').resids.shape[0]
contact_indices = np.array(list(itertools.combinations(range(nresi), 2)))

# Ca-Ca distances over the ensemble
# THIS WILL CALCULATE A SQUARE MATRIX WITH INTER-RES1UE CONTACT PROABILITIES
# SPECIFIED: CA-CA DISTANCES USED, CONTACT-CUTOFF = 10.0A
print("Load subensemble trajectory - mdtraj ...", flush=True)
model = mdj.load(str(path2traj),top=str(path2traj))

# MD ensemble --> distance matrix
print("Calculating inter-residue distances for each frame...", flush=True)
contact_indices = np.array(list(itertools.combinations(range(nresi), 2)))
distances = mdj.compute_contacts(model, contacts = contact_indices, scheme = 'ca',periodic = False)

# reshape
# Reshaped version of distances, such that the distance, in the k`th frame of the 
# trajectory from residue `i to residue j is given by contact_maps[k, i, j]. All entries 
# in contact_maps corresponding to the distance between residues that were not part of residue_pairs are 0.0.
contact_distances = mdj.geometry.squareform(distances[0],distances[1])

# convert to Angstroms
contact_distances_model = contact_distances*10

# convert to contact probability
# and save pairwise calpha contact probability map
print("Convert to contact probabilities...", flush=True)
prob = (contact_distances_model<=cutoff)*1
wopt = np.load(str(path2weights))
wopt = wopt.ravel()
prob_avg = np.average(prob,weights=wopt,axis=0)
np.save(str(path2savedcmap),prob_avg)
print("Pairwise Contact Probility Map Done!", flush=True)

# calculate the normalised sum contact probability
summed_prob = np.sum(prob_avg,axis=0)
min_sum = np.min(summed_prob)
max_sum = np.min(summed_prob)
norm = (summed_prob - min_sum) / (max_sum - min_sum)
np.save(str(path2savednormcprob),norm)

# for cross ensemble comparison of the normalised sum contact probability the summed probabilities need to be normalised across the ensemble
# therefore min and max needs to be set outside of the script as it will be the min and max across all the ensembles
min_sum = sys.argv[8] # minimum summed contact probability across all ensembles (=3.7 for AAT ensembles)
max_sum = sys.argb[9] # maximum summed contact probability across all ensembles (=26.0 for AAT ensembles)
norm = (summed_prob - min_sum) / (max_sum - min_sum)
np.save(str(path2savednormcprob),norm)

print("Normalised Sum Contact Probability Done!, flush=True)
