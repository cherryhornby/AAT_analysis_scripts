## This is a script to calculate RMSD of free energy progiles of bootstrapped samples ##
## Example command: python calc_fep_bootstrap_rmsd.py COLVAR 160 170 0.5 20 100000 ##

# import modules
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# set variables
path2plumed = sys.argv[1]
lowtemp = sys.argv[2]
hightemp = sys.argv[3]
tempstep = sys.argv[4]
num_replicas = sys.argv[5]
num_of_frames = sys.argv[6]

# Arrange arrays of CVs and weights across replicas
T = np.arange(lowtemp,hightemp,tempstep)
cvs = []
wts = []
for i in range(num_replicas):
    data = np.loadtxt(str(path2plumed))
    cv = data[:,1]
    cv = cv[len(cv)-num_of_frames:]
    logW = data[:,2]
    logW = logW[len(logW)-num_of_frames:]
    weights = np.exp((logW-np.amax(logW))/(kb*T[i]))  # unbiased reweighting factors
    cvs.append(cv)
    wts.append(weights)

# bins and blocks

bins = np.arange(0.01,0.99,0.01)   # CV bins
block_length = 50               # block length ≥ autocorrelation time
B = 200                         # number of bootstrap replicates


n = np.linspace(0,0.9,num_replicas)
cmap = mpl.colormaps['gnuplot2']
fig, ax = plt.subplots(1, 1,figsize=(6,4))

for i in range(num_replicas):
    fractions = np.linspace(0.1, 1.0, 10)  # from 10% to 100% of trajectory
    convergence_profiles = []
    
    cv = cvs[i]
    weights = wts[i]
    
    # Bootstrap profiles at each fraction
    for frac in fractions:
        nuse = int(frac * len(cv))
        cv_sub = cv[:nuse]
        weights_sub = weights[:nuse]
    
        # partition into blocks
        nblocks = nuse // block_length
        blocks = [np.arange(j*block_length, (j+1)*block_length) for j in range(nblocks)]
    
        boot_profiles = []
        for b in range(B):
            chosen_blocks = np.random.choice(nblocks, size=nblocks, replace=True)
            idx = np.concatenate([blocks[cb] for cb in chosen_blocks])
            cv_boot = cv_sub[idx]
            w_boot = weights_sub[idx]
    
            hist, edges = np.histogram(cv_boot, bins=bins, weights=w_boot, density=True)
            F = -np.log(hist + 1e-12)
            F -= np.min(F)
            boot_profiles.append(F)
    
        boot_profiles = np.array(boot_profiles)
        F_mean = np.mean(boot_profiles, axis=0)
        convergence_profiles.append(F_mean)
    
    convergence_profiles = np.array(convergence_profiles)  # shape = (nfractions, nbins)
    
    # Compute RMSD against final profile (100% data)
    F_ref = convergence_profiles[-1]  # final profile at 100%
    rmsd = np.sqrt(np.mean((convergence_profiles - F_ref) ** 2, axis=1))
    
    # Plot convergence metric
    ax.plot(fractions*100, rmsd, marker='o',color=cmap(n[i]))

ax.set_ylim(0,2)
ax.set_xlabel("Trajectory used (%)", fontsize=16)
ax.set_ylabel("RMSD to final profile (kJ mol$^-$$^1$)", fontsize=16)
ax.tick_params(labelsize=14)
norm = mpl.colors.Normalize(vmin=T[0], vmax=T[-1])
cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),ax=ax, label='K')
cbar.ax.tick_params(labelsize=16)
cbar.set_label(label='K',fontsize=16)

fig.savefig('bootstrap_profile.pdf',bbox_inches='tight')
