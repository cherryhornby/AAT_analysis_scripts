## This is a script which calculates the number of atoms (calpha-only model) within a cutoff to be used for the calulation of ln(PF)=ac*Nc to compare to PEGylation data ##
## example command: python calc_contact_number_4_PEG_PFs.py top.pdb traj.xtc surface_CG.pdb 65 14 8 0.273 ## 

# import modules
import sys
import MDAnalysis as mda
from MDAnalysis.analysis import contacts
import numpy as np

# set variables
path2top = sys.argv[1] # path to topology file of ensemble
path2traj = sys.argv[2] # path to ensemble trajectory file
path2ribo = sys.argv[3] # path to ribosome structure file
peg_site = sys.argv[4] # residue ID for position of PEGylation probe site
cutoff = sys.argv[5] # cutoff radius for contacts in angstroms
neighbour_screen = sys.argv[6] # number of neighbouring residues to be excluded from contact number calculation
scalar = sys.argv[7] # value of scalar used for ln(PF) calculation

# load universe
print("Loading trajectory...", flush=True)
u = mda.Universe(str(path2top), str(path2traj))

print("Loading ribosome atoms...", flush=True)
r = mda.Universe(str(path2ribo))

# generate selections
print("Selecting calphas...", flush=True)
labelling_site = u.select_atoms(f"resid {pegsite} and name CA")
other_ca = u.select_atoms(f"name CA and not resid {pegsite-neighbour_screen}:{pegsite+neighbour_screen}")

# function for contact number calculation
# sets probe site, atoms involved in cutoff and cutoff radius
# calculates number of calphas in contact with probe for each frame and return this in an array
def contacts_within_cutoff(u, group_a, group_b, radius=4.5):
	timeseries = []
	for ts in u.trajectory:
		# merge ribosome beads with nascent chain
		rnc = mda.Merge(group_b.atoms,r.atoms)
		group_rnc = rnc.select_atoms('all')
		# calculate distances between group_a and group_rnc
		dist = contacts.distance_array(group_a.positions, group_rnc.positions)
		# determine which distances <= radius
		n_contacts = contacts.contact_matrix(dist, radius).sum()
		timeseries.append([ts.frame, n_contacts])
	return np.array(timeseries)

# calculate contact number timeseries
print("Calculating contacts number...", flush=True)
cn = contacts_within_cutoff(u, labelling_site, other_ca, radius=14)
np.save(f'per_frame_contact_no_r{peg_site}.npy',cn)

# calculate protection factors -- ln(PF) = ac*Nc
print("Calculating ln(PF)...", flush=True)
lnPF = scalar*cn
# save as plain text file for reweighting purposes
np.savetxt(f'sim-{pegsite}C-generic.dat',lnPF)


