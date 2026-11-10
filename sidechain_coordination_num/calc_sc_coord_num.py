## this is a script that calculates the coordination number for each residue based on the pairwise distances between side chain centre of masses ##
## example command: python calc_sc_coord_num.py top.pdb traj.xtc ## 

# import modules
import sys
import mdtraj as md
import numpy as np
import pandas as pd

# set variables
path2top = sys.argv[1] # path to topology file set in command line
path2traj = sys.argv[2] # path to ensemble trajectory file set in command line
r0 = 0.45      # nm (≈ 4.5 Å)
n = 6          # exponent for switching function
exclude_seq_sep = 1  # exclude residues i ± this many
frame_stride = 1     # e.g. set to 10 to analyze every 10th frame

# define function of coordination number calculation
def switching(r, r0=r0, n=n):
    """Smooth coordination switching function."""
    return 1.0 / (1.0 + (r / r0)**n)

# load universe
print("Loading trajectory...")
t = md.load(str(path2traj), top=str(path2top))[::frame_stride]
top = t.topology
residues = list(top.residues)
nres = len(residues)
print(f"Trajectory: {t.n_frames} frames, {nres} residues")

# Compute sidechain center of mass for each residue
def residue_sidechain_coms(traj):
    residues = list(traj.top.residues)
    nres = len(residues)
    coms = np.zeros((traj.n_frames, nres, 3))
    for i, res in enumerate(residues):
        side_atoms = [a.index for a in res.atoms if a.is_sidechain and a.element.symbol != 'H']
        if not side_atoms:  # e.g., Glycine
            side_atoms = [a.index for a in res.atoms if a.name == 'CA']
        coords = traj.atom_slice(side_atoms).xyz  # (n_frames, n_atoms, 3)
        masses = np.array([traj.top.atom(idx).element.mass for idx in side_atoms])
        masses /= masses.sum()
        coms[:, i, :] = np.einsum('fai,a->fi', coords, masses)
    return coms

print("Computing sidechain centers of mass...")
coms = residue_sidechain_coms(t)

# Compute coordination numbers per frame and residue
CN_time = np.zeros((t.n_frames, nres))

print("Computing coordination numbers...")
for f in range(t.n_frames):
    # pairwise residue distances (using precomputed COMs)
    diff = coms[f, :, None, :] - coms[f, None, :, :]
    dmat = np.linalg.norm(diff, axis=2)

    for i in range(nres):
        # Exclude self and sequence neighbors
        excl_mask = np.zeros(nres, dtype=bool)
        for s in range(-exclude_seq_sep, exclude_seq_sep + 1):
            j = i + s
            if 0 <= j < nres:
                excl_mask[j] = True
        allowed = ~excl_mask
        CN_time[f, i] = np.sum(switching(dmat[i, allowed]))

# Average over frames
CN_mean = CN_time.mean(axis=0)
CN_std  = CN_time.std(axis=0)

# Output results
df = pd.DataFrame({
    "resid": [r.resSeq for r in residues],
    "resname": [r.name for r in residues],
    "CN_mean": CN_mean,
    "CN_std": CN_std
})

df.to_csv("sidechain_coordination.csv", index=False)

print("Done!")
