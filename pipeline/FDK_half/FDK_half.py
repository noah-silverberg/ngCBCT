import numpy as np
from tigre.utilities.geometry import geometry
import tigre.algorithms as algs
from tigre.utilities.filtering import filtering
from tigre.utilities.Atb import Atb
import tigre
import warnings
import copy


class FDKHalf:
    def __init__(self):
        pass
        
    def __call__(self, proj, geo, angles, **kwargs):
        """
        FDK for displaced detector (HF scan)
        This is the Python version of Half han FDK reconstruction written by Hao Zhang in Matlab
        Parameters:
        -----------
        proj : ndarray
            Projection data (nangles, vdetector, udetector)
        geo : geometry
            TIGRE geometry object
            Example:
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

        angles : ndarray
            Projection angles in radians
        **kwargs:
            filter: str, optional (not useful)
                Filter type for reconstruction (default: 'ram-lak')
            parker: bool, optional
                Apply parker weighting (default: False if scan range >= 1.5π)
        
        Returns:
        --------
        ndarray
            Reconstructed volume
        """
        

        filter_type, parker = self._parse_inputs(proj, geo, angles, kwargs)
        geo = self._check_geo(geo, angles)

        #geo.filter = filter_type
         # Check angle direction
        if angles[-1] - angles[0] < 0:
            angles = angles[::-1]
            proj = proj[:, :, ::-1]


        # Handle detector offset
        if len(geo.offDetector.shape) == 1:
            offset = np.tile(geo.offDetector[:, np.newaxis], (1, angles.shape[0]))
        else:
            offset = geo.offDetector
            
        # Zero-padding to avoid FFT-induced aliasing
        zproj, zgeo, theta = self._zero_padding(proj, geo)
        
        # Preweighting using Wang function
        proj, pre_w = self._preweighting(zproj, zgeo, theta)
        
        # Update geometry
        geo = zgeo
        
       
        # Weight
        #proj = np.transpose(proj, (1, 0, 2))
        # since all detector offsets are identical, use simple method to calculate
        # for i in range(angles.shape[0]):
        #     us = (np.arange(-geo.nDetector[0]/2 + 0.5, geo.nDetector[0]/2, 1) * 
        #           geo.dDetector[0] + offset[i, 0])
        #     vs = (np.arange(-geo.nDetector[1]/2 + 0.5, geo.nDetector[1]/2, 1) * 
        #           geo.dDetector[1] + offset[i, 1])
        #     uu, vv = np.meshgrid(us, vs)
            
        #     # Create weight according to each detector element
        #     w = geo.DSD[i] / np.sqrt(geo.DSD[i]**2 + uu**2 + vv**2)
            
        #     # Multiply weights with projection data
        #     proj[i, :, :] = proj[i, :, :] * w.T

        # Create weight according to each detector element
        us = (np.arange(-geo.nDetector[0]/2 + 0.5, geo.nDetector[0]/2, 1) * 
                  geo.dDetector[0] + offset[0, 0])
        vs = (np.arange(-geo.nDetector[1]/2 + 0.5, geo.nDetector[1]/2, 1) * 
                  geo.dDetector[1] + offset[0, 1])
        uu, vv = np.meshgrid(us, vs)     
        w = geo.DSD[0] / np.sqrt(geo.DSD[0]**2 + uu**2 + vv**2)
        #for i in range(angles.shape[0]):
                                    
            # Multiply weights with projection data
        proj[:, :, :] = proj[:, :, :] * w.T    

        # Fourier transform based filtering
        proj = filtering(proj, geo, angles, parker)
        proj=proj.astype(np.float32)
        # Remove filter field from geometry
        if hasattr(geo, 'filter'):
            delattr(geo, 'filter')
            
        # Backproject
        #res = algs.backproj(proj, geo, angles)
        res = Atb(proj, geo, angles)
        
        return res
        
    def _parse_inputs(self, proj, geo, angles, kwargs):
        """Parse input arguments"""
        if np.ptp(angles) < 1.5 * np.pi:
            warnings.warn('Parker weighting is applied for short scan reconstruction')
            parker = kwargs.get('parker', False)
        else:
            parker = False
            
        #filter_type = kwargs.get('filter', 'ram-lak')
        filter_type='ram-lak'
        filter_type = kwargs['filter'] #, 'ram-lak')
        if not isinstance(filter_type, str):
            raise ValueError('Invalid filter type')
            
        return filter_type, parker
        
    def _check_geo(self, geo, angles):
        """Verify and update geometry"""
        # Implementation depends on specific geometry checks needed
        # This is a placeholder for geometry validation
        geo.check_geo(angles)
        return geo
        
    def _zero_padding(self, proj, geo):
        """Zero padding preprocessing"""
        zgeo = copy.deepcopy(geo)
 
        padwidth = int(2 * geo.offDetector[0][1] / geo.dDetector[1])
        abs_padwidth=abs(padwidth)
        zgeo.offDetector=zgeo.offDetector.astype(np.float32)
        zgeo.offDetector[:, 1] = geo.offDetector[:, 1] - padwidth/2 * geo.dDetector[1]
        zgeo.nDetector[1] = abs(padwidth) + geo.nDetector[1]
        zgeo.sDetector[1] = zgeo.nDetector[1] * zgeo.dDetector[1]
        
        theta = (geo.sDetector[1]/2 - abs(geo.offDetector[0][1])) * np.sign(geo.offDetector[0][1])
        
        zproj = np.zeros((proj.shape[0], proj.shape[1], proj.shape[2]+padwidth), dtype=proj.dtype)
        if geo.offDetector[0][1] > 0:
        # Use slicing for faster assignment instead of looping
            zproj[:, :, abs_padwidth:] = proj
        else:
            zproj[:, :, :proj.shape[2]] = proj
        
        # if geo.offDetector[0][1] > 0:
        #     for i in range(proj.shape[0]):
        #         zproj[i, :, :] = np.hstack([np.zeros((proj.shape[1], padwidth)), proj[i, :, :]])
        # else:
        #     for i in range(proj.shape[0]):
        #         zproj[i, :, :] = np.hstack([proj[i, :, :], np.zeros((proj.shape[1], abs(padwidth)))])
                
        return zproj, zgeo, theta
        
    def _preweighting(self, proj, geo, theta):
        """Preweighting using Wang function"""
        offset = geo.offDetector[0][1]
        us = (np.arange(-geo.nDetector[1]/2 + 0.5, geo.nDetector[1]/2, 1) * 
              geo.dDetector[1] + abs(offset))
        
        abstheta = abs(theta)
        w = np.ones(proj[0, :, :].shape)

         # Vectorized implementation of the loop logic
        mask1 = np.abs(us) <= abstheta
        w[:, mask1] = 0.5 * (np.sin((np.pi/2) * np.arctan(us[mask1]/geo.DSD[0]) /
                            (np.arctan(abstheta/geo.DSD[0]))) + 1)

        mask2 = us < -abstheta
        w[:, mask2] = 0
        
        # for i in range(int(geo.nDetector[1])):
        #     t = us[i]
        #     if abs(t) <= abstheta:
        #         w[:, i] = 0.5 * (np.sin((np.pi/2) * np.arctan(t/geo.DSD[0]) / 
        #                               (np.arctan(abs(theta)/geo.DSD[0]))) + 1)
        #     if t < -abstheta:
        #         w[:, i] = 0
                
        if theta < 0:
            w = np.fliplr(w,axis=1)
            
        proj_w = np.zeros_like(proj)
        
        proj_w[:, :, :] = proj[:, :, :] * w * 2

        # for i in range(proj.shape[0]):
        #     proj_w[i, :, :] = proj[i, :, :] * w * 2
            
        return proj_w, w