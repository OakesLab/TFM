'''
Functions for Traction Force Microscopy analysis

'''

import os                                                      # making and managing directories
import glob as glob                                            # grabbing file names
import pandas as pd                                            # making dataframe for exporting parameters
import numpy as np                                             # basic math
import skimage.io as io                                        # reading in images
from scipy.ndimage import morphology                           # morphological operations
from skimage.measure import label, regionprops                 # image comparison tools
from skimage.morphology import opening, disk, remove_small_objects, remove_small_holes    # image filtering operations
import cv2                                                     # for filtering vector fields
import matplotlib.pyplot as plt                                # for plotting
from matplotlib import cm, colors                              # for controlling colormaps
from TFM_FTTC_tools import *



def TFM_calculation(shear_modulus = 8600, um_per_pixel = .103174, regparam = 1e-16, downsample = 12, timepoint = 1, check_figure=False, fig_max_stress = 1000):
    # Find the displacement files
    file_list_dispu = sorted(glob.glob('displacement_files/disp_u*.tif'))
    file_list_dispv = sorted(glob.glob('displacement_files/disp_v*.tif'))

    # Number of files to process
    N_images = len(file_list_dispu)

    temp_image = io.imread(file_list_dispu[0])
    N_rows, N_cols = temp_image.shape

    # Define the parameters
    E = shear_modulus * 3
    nu = 0.5
    bead_depth = 0
    alpha = 0
    pad_fraction = 0
    lanczos_exp = 1
    mesh_size = 1

    # for loop to go over all the images in the stack

    # make a directory to store all the traction files
    if os.path.isdir('traction_files/') == False:
        os.mkdir('traction_files/')

    # make a directory to store all the displacement files
    if os.path.isdir('recovered_displacement_files/') == False:
        os.mkdir('recovered_displacement_files/')

    # make an empty image stack to hold traction  and recovered displacement maps
    traction_maps = np.zeros((N_images,N_rows, N_cols))
    displacement_maps = np.zeros((N_images,N_rows, N_cols))
    #initial things that will be needed for all planes

    # make matrices with the appropriate x and y coordinates
    x = np.arange(0, N_cols, mesh_size)
    y = np.arange(0, N_rows, mesh_size)
    X, Y = np.meshgrid(x,y)
    grid_mat = np.zeros((2,N_rows,N_cols))
    grid_mat[0,:,:] = X
    grid_mat[1,:,:] = Y
    i_bound_size = 0
    j_bound_size = 0
    i_max = grid_mat.shape[1]
    j_max = grid_mat.shape[2]


    # make an array to hold the regularization parameter
    lam = np.zeros(N_images)

    # calculate the fourier modes
    kx, ky, lanczosx, lanczosy = calculate_fourier_modes(mesh_size, i_max, j_max, lanczos_exp)
    # calculate the Green's Function
    GFt = calculate_greens_function(E, nu, kx, ky, i_max, j_max, mesh_size, bead_depth)

    for t in np.arange(0,N_images):
        # read in strain fields
        disp_u = io.imread(file_list_dispu[t])
        disp_v = io.imread(file_list_dispv[t])

        # put strain field into proper matrix
        u = np.zeros((2,N_rows,N_cols))
        u[0,:,:] = disp_u
        u[1,:,:] = disp_v

        # Calculate the inverse Green's function
        G_inv_xx, G_inv_xy, G_inv_yy = calculate_Ginv(GFt, regparam)

        # Perform the actual TFM
        Ftfx, Ftfy = reg_fourier_TFM_L2(u, G_inv_xx, G_inv_xy, G_inv_yy)
        # Reshape tractions in frequency space into an array
        Ftf = np.array([Ftfx, Ftfy])
        # Recover the displacement field from the traction stresses
        urec, Fturec = reconstruct_displacement_field(GFt, Ftfx, Ftfy, lanczosx, lanczosy)
        # Recover the actual traction stresses
        pos, vec, fnorm, f, energy = calculate_stress_field(Ftfx, Ftfy, lanczosx, lanczosy, grid_mat, u, i_max, j_max,
                                                        i_bound_size, j_bound_size, um_per_pixel, mesh_size)

        # store the norm in our traction map stack
        traction_maps[t,:,:] = fnorm.copy()
        # save the traction stresses as images
        io.imsave('traction_files/fx_%03d.tif' % (t + 1), f[0,:,:].astype('float32'), check_contrast=False)
        io.imsave('traction_files/fy_%03d.tif' % (t + 1), f[1,:,:].astype('float32'), check_contrast=False)

        # store the recovered displacement norm
        urec_norm = np.sqrt(urec[0,:,:]**2 + urec[1,:,:]**2)
        displacement_maps[t,:,:] = urec_norm.copy()
        # save the recovered displacement vectors as images
        io.imsave('recovered_displacement_files/disp_ur_%03d.tif' % (t + 1), urec[0,:,:].astype('float32'), check_contrast=False)
        io.imsave('recovered_displacement_files/disp_vr_%03d.tif' % (t + 1), urec[1,:,:].astype('float32'), check_contrast=False)

    io.imsave('traction_maps.tif',traction_maps.astype('int16'), check_contrast=False)
    io.imsave('recovered_displacement_maps.tif',displacement_maps.astype('float32'), check_contrast=False)

    # Create a dictionary of our parameters
    TFM_params = {
        "shear_modulus" : shear_modulus,
        "E" : E,
        "nu" : nu,
        "um_per_pixel" : um_per_pixel,
        "bead_depth" : bead_depth,
        "alpha" : alpha,
        "pad_fraction" : pad_fraction,
        "lanczos_exp" : lanczos_exp,
        "ref_image_rows" : N_rows,
        "ref_image_cols" : N_cols,
        "mesh_size" : mesh_size,
        "f_rows" : fnorm.shape[0],
        "f_cols" : fnorm.shape[1],
        }
    # Convert the dictionary to a DataFrame
    TFM_params_df = pd.DataFrame(TFM_params, index=[0])
    # Write the parameters to a CSV file for saving
    TFM_params_df.to_csv('TFM_params.csv')

    # save as a .txt file the regularization paramers
    np.savetxt('Regularization_parameter.txt',[regparam])
    
    # for saving an image

    # Find the displacement and traction stress files
    file_list_fx = sorted(glob.glob('traction_files/fx_*.tif'))
    file_list_fy = sorted(glob.glob('traction_files/fy_*.tif'))
    fx = io.imread(file_list_fx[timepoint - 1])
    fy = io.imread(file_list_fy[timepoint - 1])
    fnorm = np.sqrt(fx**2 + fy**2)

    x_small = grid_mat[0,::downsample,::downsample]
    y_small = grid_mat[1,::downsample,::downsample]
    fx_small = fx[::downsample,::downsample]
    fy_small = fy[::downsample,::downsample]


    traction_map_fig, traction_map_axes = plt.subplots()
    traction_map_axes.imshow(fnorm, vmin=0, vmax=np.max(fnorm)*.9)
    traction_map_axes.quiver(x_small, y_small, fx_small, -fy_small, color='w', alpha=0.5)
    #traction_map_fig.savefig('myimage.svg', format='svg', dpi=1200)
    traction_map_fig.savefig('Traction_vectors.png', format='png', dpi=300)

    if check_figure == True:
        urec_norm = np.sqrt(urec[0,:,:]**2 + urec[1,:,:]**2)
        # Display the Displacement maps and Traction Maps
        TFM_check_fig, TFM_check_axes = plt.subplots(nrows=2, ncols=4)
        #TFM_check_axes[0,0].imshow(unorm, vmin=0, vmax=3)
        #TFM_check_axes[0,0].set_title('Displacement\nMap')
        #TFM_check_axes[0,0].axis('off')
        TFM_check_axes[0,0].imshow(urec_norm, vmin=0, vmax=np.max(urec_norm)*.9)
        TFM_check_axes[0,0].set_title('Recovered\nDisplacement\nMap')
        TFM_check_axes[0,0].axis('off')
        TFM_check_axes[0,1].imshow(fnorm, vmin=0, vmax=np.max(fnorm)*.9)
        TFM_check_axes[0,1].set_title('Tractions\nStresses')
        TFM_check_axes[0,1].axis('off')
        TFM_check_axes[0,2].imshow(f[0,:,:] )
        TFM_check_axes[0,2].set_title('FX Traction\nStresses')
        TFM_check_axes[0,2].axis('off')
        TFM_check_axes[0,3].imshow(f[1,:,:])
        TFM_check_axes[0,3].set_title('FY Traction\nStresses')
        TFM_check_axes[0,3].axis('off')
    
        TFM_check_axes[1,0].imshow(grid_mat[0,:,:])
        TFM_check_axes[1,0].set_title('x\ncoordinates')
        TFM_check_axes[1,0].axis('off')
        TFM_check_axes[1,1].imshow(grid_mat[1,:,:])
        TFM_check_axes[1,1].set_title('y\ncoordinates')
        TFM_check_axes[1,1].axis('off')
        TFM_check_axes[1,2].imshow(u[0,:,:])
        TFM_check_axes[1,2].set_title('u\ncoordinates')
        TFM_check_axes[1,2].axis('off')
        TFM_check_axes[1,3].imshow(u[1,:,:])
        TFM_check_axes[1,3].set_title('v\ncoordinates')
        TFM_check_axes[1,3].axis('off')
        TFM_check_fig.tight_layout()
        TFM_check_fig.show()
    return 


def TFM_analysis(GFP='', force_min=0):

    # Read in the CSV file with all the file details and convert to a dictionary
    temp_dict = pd.read_csv('TFM_params.csv')
    TFM_params = {}
    for key in temp_dict.keys()[1:]:
        TFM_params[key] = temp_dict[key][0]

    # Find the displacement and traction stress files
    file_list_fx = sorted(glob.glob('traction_files/fx_*.tif'))
    file_list_fy = sorted(glob.glob('traction_files/fy_*.tif'))
    file_list_dispu = sorted(glob.glob('displacement_files/disp_u*.tif'))
    file_list_dispv = sorted(glob.glob('displacement_files/disp_v*.tif'))

    # Number of files to process
    N_images = len(file_list_fx)

    # read in the cell mask
    cellmask = io.imread('cellmask.tif').astype(bool)
    # read in the mask for the forces
    forcemask = io.imread('forcemask.tif').astype(bool)
    # check if it's a single plane or a stack
    if len(cellmask.shape) == 2:
        cellmask_stack = np.zeros((N_images,cellmask.shape[0],cellmask.shape[1]))
        cellmask_stack[:] = cellmask
        cellmask = cellmask_stack.astype(bool)
    if len(forcemask.shape) == 2:
        forcemask_stack = np.zeros((N_images,forcemask.shape[0],forcemask.shape[1]))
        forcemask_stack[:] = forcemask
        forcemask = forcemask_stack.astype(bool)

    if len(GFP) > 0:
        # set GFP image flag to True
        GFP_image = True
        # read in the GFP channel
        cell_image = io.imread(GFP).astype('float')
        # check if it's a single plane or a stack
        if len(cell_image.shape) == 2:
            cell_image = np.expand_dims(cell_image, axis=0)
    else:
        GFP_image = False
    
    # create empty variables to store all the data
    energy = []
    energy_per_area = []
    residual = []
    force_sum = []
    displacement_sum = []
    time = []
    cell_area = []
    if GFP_image:
        GFP_mean_intensity = []
        GFP_sum_intensity = []

    # loop over all the frames in the series

    for timepoint in np.arange(0,N_images):

        # read in the file
        tractionx = io.imread(file_list_fx[timepoint])
        tractiony = io.imread(file_list_fy[timepoint])
        dispx = io.imread(file_list_dispu[timepoint])
        dispy = io.imread(file_list_dispv[timepoint])

        # only use points in the forcemask
        tractionx = tractionx[forcemask[timepoint]]
        tractiony = tractiony[forcemask[timepoint]]
        dispx = dispx[forcemask[timepoint]]
        dispy = dispy[forcemask[timepoint]]

        if force_min > 0:
            # calculate the magnitude at each pixel
            traction_mag = np.sqrt(tractionx**2 + tractiony**2)
            traction_thresh_mask = traction_mag > force_min
            tractionx = tractionx[traction_thresh_mask]
            tractiony = tractiony[traction_thresh_mask]
            dispx = dispx[traction_thresh_mask]
            dispy = dispy[traction_thresh_mask]


        # energy is one half the sum of the dot product of the traction vector with the displacement vector
        # need to include corrections for the units and the area covered. 10^-6 is to put the number in pJ
        energy.append( 0.5 * np.sum(((dispx * tractionx) + dispy * tractiony)) * TFM_params['mesh_size']**2 * TFM_params['um_per_pixel']**3 * 10**-6)

        # force_sum is the sum of the absolute magnitudes of the vectors in the mask
        force_sum.append( np.sum( np.sqrt(tractionx**2 + tractiony**2)))

        # sum of the displacement magnitudes
        displacement_sum.append( np.sum( np.sqrt(dispx**2 + dispy**2)))
        
        # residual is an error metric. Should be less than 0.1 (e.g. 10%)
        residual.append( np.sqrt( np.sum(tractionx) ** 2 + np.sum(tractiony) ** 2) / force_sum[timepoint] * 100)
        
        # stores the time point
        time.append(timepoint)
        # calculate cell area
        cell_area.append(np.sum(cellmask[timepoint]) * (TFM_params['um_per_pixel'] ** 2))
        # calculate energy per area
        energy_per_area.append(energy[timepoint] / cell_area[timepoint])
        # store the force_min
        if GFP_image:
            # calculate GFP mean intensity and sum intensity
            GFP_mean_intensity.append(np.mean(cell_image[timepoint][cellmask[timepoint]]))
            GFP_sum_intensity.append(np.sum(cell_image[timepoint][cellmask[timepoint]]))

    # Convert the lists of data to a dictionary and save it as a CSV file
    TFM_analysis_dict = {
        'time': time,
        'cell_area_microns2' : cell_area,
        'force_minimum' : [force_min] * N_images,
        'force_sum_Pa': force_sum,
        'displacement_sum': displacement_sum,
        'residual': residual,
        'energy_pJ': energy,
        'energy_per_area': energy_per_area
        }

    if GFP_image:
        TFM_analysis_dict['mean_GFP_intensity'] = GFP_mean_intensity
        TFM_analysis_dict['sum_GFP_intensity'] = GFP_sum_intensity

    # TFM_analysis_dict['energy_pJ']
    TFM_dataframe = pd.DataFrame(TFM_analysis_dict)
    TFM_dataframe.head(10)
    TFM_dataframe.to_csv('TFM_analysis.csv')

    return

def cellmask_threshold(imagename, small_object_size=50, cell_minimum_area=50000, dilation_size = 10, save_figure=True, plot_figure=True, timepoint = 0):
    # check if it's a string or a matrix and read in the image
    if isinstance(imagename, str):
        imagestack = io.imread(imagename, plugin='tifffile', is_ome=False)
    else:
        imagestack = imagename
    
    # determine the number of planes
    if len(imagestack.shape) == 2:
        imagestack = np.expand_dims(imagestack, axis=0)
    
    # create empty matrices to hold our masks
    cellmask_stack = np.zeros(imagestack.shape)
    forcemask_stack = np.zeros(imagestack.shape)
    
    # loop through each image in the stack
    for plane, image in enumerate(imagestack):
        # Find the unique intensity values in the image
        intensity_values = np.unique(image.ravel())

        # reduce list of intensity values down to something manageable to speed up computation
        slice_width = np.round(len(intensity_values)/300).astype('int')
        # if the intesnity spread is less than 150 you need to correct the rounding to be greater than 1
        if slice_width == 0:
            slice_width = 1
        intensity_values = intensity_values[::slice_width]

        # Find the mean intensity value of the image
        intensity_mean = np.mean(image)

        # create a zero matrix to hold our difference values
        intensity_difference = np.zeros_like(intensity_values).astype('float')

        # for loop to compare the difference between the intensity sum of pixels above a threshold 
        # and the average image intensity of an identical number of pixels
        for i,intensity in enumerate(intensity_values):
            # make a mask of pixels about a given intensity
            mask = image > intensity

            # take the difference between the sum of thresholded pixels and the average value of those pixels
            intensity_difference[i] = np.sum(mask * image) - intensity_mean*np.sum(mask)

        # find the maximum value of the intensity_difference and set it equal to the threshold
        max_intensity = np.argwhere(intensity_difference == np.max(intensity_difference))
        threshold = intensity_values[max_intensity[0][0]]

        # make a mask at this threshold
        mask = image > threshold

        # get rid of small objects
        mask = remove_small_objects(mask, small_object_size)
        # fill any holes in the mask
        mask = morphology.binary_fill_holes(mask)
        # remove anything on the border

        # label the mask objects and get region props
        mask_label = label(mask)
        props = regionprops(mask_label)

        # define the center of the image coordinates
        center_row = image.shape[0]/2
        center_col = image.shape[1]/2

        # find the largest and closest object and keep only that one
        areas = []
        distance = []
        for region in props:
            areas.append(region.area)
            # calculate the distance from the center of the image to the object centroid
            distance.append(np.sqrt((center_row - region.centroid[0])**2 + (center_col - region.centroid[1])**2))

        # create a loop to check on if the object is centered and large
        check_point = True
        # list of labels to keep track of the object in questions
        label_list = list(range(len(areas)))
        while check_point == True:
            # find the object with the closest centroid
            closest_region = np.argwhere(distance == np.min(distance))
            # remove that object from the list
            distance.pop(closest_region[0][0])
            # remove that label from the list
            closest_region_label = label_list.pop(closest_region[0][0])
            # check if that object has an area greater than our cell minimum area 
            if areas[closest_region_label] > cell_minimum_area:
                break

        # use that label to make your cell mask
        cellmask = mask_label == (closest_region_label + 1)

        # make a structuring element to filter the binary image with
        #SE2 = disk(5)
        # filter the original image
        #cellmask = cv2.dilate(firstpass_mask.astype('uint8'), SE2)
        # fill any holes
        #cellmask = morphology.binary_fill_holes(cellmask)

        # make a forcemask
        forceSE = disk(dilation_size)
        forcemask = cv2.dilate(cellmask.astype('uint8'), forceSE)

        cellmask_stack[plane] = cellmask.copy()
        forcemask_stack[plane] = forcemask.copy()



    # plot figure 
    if plot_figure == True:
        cwd = os.getcwd()
        cellname = cwd[cwd.find('cell'):] 
        # check to see if tractionmaps exist
        traction_file = glob.glob('traction_maps.tif')
        if len(traction_file) == 1:
            tractionmap = io.imread('traction_maps.tif')
            if len(tractionmap.shape) > 2:
                tractionmap = tractionmap[timepoint]
        else:
            tractionmap = np.zeros_like(forcemask)

        # plotting for data confirmation
        mask_fig, mask_axes = plt.subplots(nrows=2, ncols=2)
        mask_axes[0,0].imshow(imagestack[timepoint], cmap='Greys_r', vmin=np.min(image), vmax=np.max(image)*.8)
        mask_axes[0,0].set_title(cellname)
        mask_axes[0,1].imshow(imagestack[timepoint], cmap='Greys_r', vmin=np.min(image), vmax=np.max(image)*.8)
        mask_axes[0,1].imshow(cellmask_stack[timepoint], alpha=0.2)
        mask_axes[0,1].set_title('cellmask')
        mask_axes[1,0].imshow(tractionmap, vmin = 0, vmax = np.max(tractionmap)*.8)
        mask_axes[1,1].imshow(tractionmap, vmin = 0, vmax = np.max(tractionmap)*.8)
        mask_axes[1,1].imshow(forcemask_stack[timepoint], alpha=0.2)
        mask_axes[1,1].set_title('forcemask')
        mask_axes[0,0].axis('off')
        mask_axes[1,0].axis('off')
        mask_axes[0,1].axis('off')
        mask_axes[1,1].axis('off')
        mask_fig.tight_layout()
        mask_fig.savefig('mask_check.png', format='png', dpi=300)



    if save_figure == True:
        io.imsave('cellmask.tif', cellmask_stack.astype('uint8') * 255, check_contrast=False)
        io.imsave('forcemask.tif', forcemask_stack.astype('uint8') * 255, check_contrast=False)

    # reduce dimensions on stack if only one plane
    if cellmask_stack.shape[0] == 1:
        cellmask_stack = cellmask_stack[0]
        forcemask_stack = forcemask_stack[0]

    return cellmask_stack, forcemask_stack, threshold


def crop_TFM_image(frame, width, height, corner = (0,0), mask = None, arrow_spacing = 12, LUT = 'viridis', arrow_scale = None, 
                   TFM_min = 0, TFM_max = None, arrow_color = 'w', min_arrow_mag = None, arrow_width = 1, colorbar = False, 
                   save_fig = True, file_type = 'both', dpi=150, base_folder = '', show_fig = True):
    # get list of images
    fx_im_list = sorted(glob.glob('traction_files/fx*.tif'))
    fy_im_list = sorted(glob.glob('traction_files/fy*.tif'))
    # read in the frame of interest
    fx = io.imread(fx_im_list[frame])
    fy = io.imread(fy_im_list[frame])
    
    # crop fx and fy
    fx_cropped = fx[corner[0]:corner[0]+height, corner[1]:corner[1]+width]
    fy_cropped = fy[corner[0]:corner[0]+height, corner[1]:corner[1]+width]
    
    # calcualte the magnitude
    TFM_mag_cropped = np.sqrt(fx_cropped**2 + fy_cropped**2)

    # set the max magnitude of the image if it isn't given
    if TFM_max is None:
        TFM_max = np.max(TFM_mag_cropped)

    # make x,y coordinates for the image
    x_cropped, y_cropped = np.meshgrid(np.arange(0,fx_cropped.shape[1]), np.arange(0,fx_cropped.shape[0]))

    # splice vectors based on spacing given
    fx_cropped = fx_cropped[::arrow_spacing,::arrow_spacing]
    fy_cropped = fy_cropped[::arrow_spacing,::arrow_spacing]
    x_cropped = x_cropped[::arrow_spacing,::arrow_spacing]
    y_cropped = y_cropped[::arrow_spacing,::arrow_spacing]
    
    if min_arrow_mag is not None:
        TFM_mag_spliced = np.sqrt(fx_cropped**2 + fy_cropped**2)
        TFM_mask = TFM_mag_spliced >= min_arrow_mag
        fx_cropped = fx_cropped[TFM_mask]
        fy_cropped = fy_cropped[TFM_mask]
        x_cropped = x_cropped[TFM_mask]
        y_cropped = y_cropped[TFM_mask]
    
    # make the figure
    plt.figure()
    # plot the magnitude image
    plt.imshow(TFM_mag_cropped, cmap = LUT, vmin = TFM_min, vmax = TFM_max)
    if colorbar:
        plt.colorbar(norm = colors.Normalize(vmin=TFM_min, vmax=TFM_max), cmap = LUT, label = 'Traction Stress (Pa)')

    # plot the vectors (fy is negative because origin is flipped for an image)
    plt.quiver(x_cropped,y_cropped,fx_cropped,-fy_cropped,color = arrow_color, scale_units='inches', 
               units = 'x', width = arrow_width, scale=arrow_scale)
    # turn the axis labels off
    plt.axis('off')



    # save the figure
    if save_fig:
        if file_type == 'PNG' or file_type == 'both':
            plt.savefig(base_folder + 'traction_maps_%03d_cropped.png' % frame, dpi = dpi, bbox_inches='tight', pad_inches = 0)
        if file_type == 'EPS' or file_type == 'both':
            plt.savefig(base_folder + 'traction_maps_%03d_cropped.eps' % frame, dpi = dpi, bbox_inches='tight', pad_inches = 0)

    if show_fig:
        plt.show()
    else:
        plt.close()
    return

    
