import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ipywidgets import interact, IntSlider
import ipywidgets as widgets

def display_3d_array_views(data_3d):
    """
    Display three orthogonal views of a 3D array with interactive slice selection.
    
    Parameters:
    data_3d : 3D numpy array
        The data to be visualized
    """
    # Create figure and subplots
    fig = plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    
    # Get initial middle slices
    slices = [s//2 for s in data_3d.shape]
    
    # Create initial plots
    im1 = ax1.imshow(data_3d[slices[0], :, :],cmap='grey')
    im2 = ax2.imshow(data_3d[:, slices[1], :],cmap='grey')
    im3 = ax3.imshow(data_3d[:, :, slices[2]],cmap='grey')
    
    # Set titles
    ax1.set_title(f'X Slice (Index: {slices[0]})')
    ax2.set_title(f'Y Slice (Index: {slices[1]})')
    ax3.set_title(f'Z Slice (Index: {slices[2]})')
    
    # Add color bars
    for ax, im in zip([ax1, ax2, ax3], [im1, im2, im3]):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(im, cax=cax)
    
    # Add sliders
    slider_ax1 = plt.axes([0.15, 0.02, 0.15, 0.03])
    slider_ax2 = plt.axes([0.45, 0.02, 0.15, 0.03])
    slider_ax3 = plt.axes([0.75, 0.02, 0.15, 0.03])
    
    s1 = Slider(slider_ax1, 'X Slice', 0, data_3d.shape[0]-1, 
                valinit=slices[0], valstep=1)
    s2 = Slider(slider_ax2, 'Y Slice', 0, data_3d.shape[1]-1, 
                valinit=slices[1], valstep=1)
    s3 = Slider(slider_ax3, 'Z Slice', 0, data_3d.shape[2]-1, 
                valinit=slices[2], valstep=1)
    
    # Update functions for sliders
    def update_x(val):
        im1.set_array(data_3d[int(val), :, :])
        ax1.set_title(f'X Slice (Index: {int(val)})')
        fig.canvas.draw_idle()
        
    def update_y(val):
        im2.set_array(data_3d[:, int(val), :])
        ax2.set_title(f'Y Slice (Index: {int(val)})')
        fig.canvas.draw_idle()
        
    def update_z(val):
        im3.set_array(data_3d[:, :, int(val)])
        ax3.set_title(f'Z Slice (Index: {int(val)})')
        fig.canvas.draw_idle()
    
    # Connect sliders to update functions
    s1.on_changed(update_x)
    s2.on_changed(update_y)
    s3.on_changed(update_z)
    
    plt.tight_layout()
    plt.show()

def display_3d_array_views_jupyter(data_3d):
    def view_slices(x_slice, y_slice, z_slice):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
        
        im1 = ax1.imshow(data_3d[x_slice, :, :],cmap='grey')
        im2 = ax2.imshow(data_3d[:, y_slice, :],cmap='grey')
        im3 = ax3.imshow(data_3d[:, :, z_slice],cmap='grey')
        
        ax1.set_title(f'X Slice (Index: {x_slice})')
        ax2.set_title(f'Y Slice (Index: {y_slice})')
        ax3.set_title(f'Z Slice (Index: {z_slice})')
        
        plt.colorbar(im1, ax=ax1)
        plt.colorbar(im2, ax=ax2)
        plt.colorbar(im3, ax=ax3)
        
        plt.tight_layout()
        plt.show()
    
    x_slider = IntSlider(min=0, max=data_3d.shape[0]-1, value=data_3d.shape[0]//2, description='X Slice')
    y_slider = IntSlider(min=0, max=data_3d.shape[1]-1, value=data_3d.shape[1]//2, description='Y Slice')
    z_slider = IntSlider(min=0, max=data_3d.shape[2]-1, value=data_3d.shape[2]//2, description='Z Slice')
    
    interact(view_slices, x_slice=x_slider, y_slice=y_slider, z_slice=z_slider)

# Example usage with random data
# if __name__ == "__main__":
#     # Create sample 3D data (50x50x50 array)
#     data = np.random.rand(50, 50, 50)
    
#     # Add some structure to make visualization more interesting
#     x, y, z = np.meshgrid(np.linspace(-2, 2, 50),
#                          np.linspace(-2, 2, 50),
#                          np.linspace(-2, 2, 50))
#     data += np.exp(-(x**2 + y**2 + z**2))
    
#     # Display the interactive visualization
#     display_3d_array_views(data)