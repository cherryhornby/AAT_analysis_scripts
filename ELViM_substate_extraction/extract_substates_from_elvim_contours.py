## this is a script that based off the free energy surface contour lines indices from the elvim projection can be selected as substates from the lowest free energy minima ##
## example command: python extract_substates_from_elvim_contours.py output.dat RG.elvim 

# import modules
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mplPath
from matplotlib import ticker, cm
import matplotlib as mpl
from scipy import stats
from scipy.ndimage.filters import gaussian_filter

# set variable
path2elvimproj = sys.argv[1]
path2rg = sys.argv[2]

# load elvim projection
x, y = np.genfromtxt(str(path2elvimproj), unpack='True') # extract X and Y from the projection file
xy=np.vstack([x,y]) # Concatenate the sequences vertically
xedges = np.arange(0,max(x)+0.1,0.01) # define bins
yedges = np.arange(0,max(y)+0.1,0.01) # define bins

# load weights
# elvim was done a subset of frames from each replica ensemble (top 95%)
# so need to be stack into one array to be used with the elvim indices
weights_arr = []
for i in range(1,4):
	w = np.load('replica{}_top_95%_weights.npy'.format(i))
	weights_arr.append(w)

combined_wopt = np.concatenate(weights_arr)

# projection ---> free energy surface
H, xedges, yedges = np.histogram2d(x,y,bins=(xedges,yedges),weights=combined_wopt)
H = gaussian_filter(H, sigma=2)
h = H/np.sum(H)
f = -0.008314*162.5*np.log(h)
f = f-np.nanmin(f)
f = f.T

# plot free energy surface
fig, ax = plt.subplots(figsize=(6,6)) # Creates the figure
plt.gca().set_aspect('equal') # Set the aspect of the axis scaling

plt.axis('off') # Removes the x and y axis
filtered_arr=gaussian_filter(f, sigma=0.1)
cs = ax.contour(f,levels = np.arange(0,6,1),colors = np.repeat(str('black'),17),
           linewidths = np.repeat(0.5,17), linestyles = np.repeat(str('dashed'),17))
cf = ax.contourf(f,levels = np.arange(0,6,1),linewidths = np.repeat(0.5,17),cmap='GnBu_r')
cbar=fig.colorbar(cf,ax=ax, pad=0.05, shrink=1)
cbar.set_label('kJ mol$^-$$^1$', fontsize=16) 
cbar.ax.tick_params(labelsize=16)
plt.savefig('elvim_fes.png', format='png', dpi=600, bbox_inches = 'tight', pad_inches = 0.05)

# load Rg data to calculate Rgs for substates
data = np.loadtxt(str(path2rg),skiprows=1)
rg = data[:,1] # plumed file with second column as Rg values for each frame

avg_rg = []
std_rg = []

# extract inidices within contours plotted 
for i in range(0,len(cs.collections[1].get_paths())):
	# contour path that represents < 1 kj/mol as plotted above to draw polygons to use as masks for vertices selection
	p = cs.collections[1].get_paths()[i]
	v = p.vertices
	cx = v[:,0]
	cy = v[:,1]
	poly = np.array(list(zip(cx,cy)))
	bbPath = mplPath.Path(poly)
	points = np.array(list(zip(x1*100,y1*100)))
	mask = bbPath.contains_points(points)
	masked_scatter = points[mask]
	frames = np.argwhere(mask==True)
	subw=combined_wopt[frames] # weights of substate vertices
	np.save('min{}_weights.npy'.format(i), subw) # save substate weights
	ar = np.average(rg[frames])
	avg_rg.append(np.average(rg[frames]))
	std_rg.append(math.sqrt(np.average((rg[frames]-ar)**2)))
	print('substate {} has {} frames and an RG of {:.2f} '.format(i,len(frames),np.average(rg[frames]))+u"\u00B1 {:.2f} nm".format(np.std(rg[frames])))
    
	# save frames from substates
	frames += 1
	f = open("min{}.ndx".format(i), "w")
	f.write('[ frames ]\n')
	for item in frames:
		f.write(str(item[0])+"\n")
	f.close()

print("Done!")
