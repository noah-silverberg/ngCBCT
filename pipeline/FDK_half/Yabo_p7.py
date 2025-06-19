import tigre
import numpy as np
#from matplotlib import pyplot as plt
from scipy.io import loadmat
from FDK_half import FDKHalf
from view3D import display_3d_array_views

## Define TIGRE Geometry
geo = tigre.geometry_default(high_resolution=True)
geo.nDetector=np.array([384,512])
geo.dDetector=np.array([0.776,0.776])
geo.sDetector=np.array([x*y for x,y in zip(geo.nDetector,geo.dDetector)])
geo.nVoxel=np.array([220,450,450])
geo.dVoxel=np.array([1,1,1])
geo.sVoxel=np.array([x*y for x,y in zip(geo.nVoxel,geo.dVoxel)])
geo.offDetector=np.array([-2,148])
geo.DSD=1500
geo.accuracy=0.5

# Load prjection data
path="\\\\pisidsmph\LI_AI_DATA\Yabo\SPARE\\raw\MonteCarloDatasets\MC_V_P7_NS_01.mat"

rawdata = loadmat(path)['data']
projs=rawdata['projs'][0][0]
projs=np.transpose(projs,(2,1,0))
# -90 degree for the data from Yabo
angles=np.squeeze(rawdata['angles'][0][0])-90.0
anglesrad = np.deg2rad(angles)

# Regular FDK
fdk=tigre.algorithms.fdk(projs,geo,anglesrad)

# FKD_half fan reconstruction. Modification from Hao's code in Matlab
# Initialize reconstructor
fdkhalf = FDKHalf()
# Reconstruct
fdk_hlaf= fdkhalf(projs, geo, anglesrad, filter='hann', parker=True)

#display results
display_3d_array_views(fdk_hlaf)