
import numpy as np                                             # basic math
import cv2                                                     # for optical flow
import skimage.io as io                                        # reading in images
import pandas as pd                                            # for parameter file
import glob as glob                                            # for finding files
import os                                                      # for making directory to store displacement files


def TFM_optical_flow(pyr_scale = 0.25, levels = 4, winsize = 24, iterations = 4, poly_n = 7, poly_sigma = 1.25):
    # Create a dictionary of our PIV parameters
    PIV_params = {
    "method" : 'Farneback Optical Flow',
    "pyr_scale" : pyr_scale,
    "levels" : levels,
    "winsize" : winsize,
    "iterations" : iterations,
    "poly_n" : poly_n,
    "poly_sigma" : poly_sigma
    }
    
    # Convert the dictionary to a DataFrame
    PIV_params_df = pd.DataFrame(PIV_params, index=[0])
    # Write the parameters to a CSV file for saving
    PIV_params_df.to_csv('PIV_params.csv')

    # Run a for loop with all the images

    # read in reference image
    ref_file_list = glob.glob('*_reference.tif')
    reference_image = io.imread(ref_file_list[0])

    # make a directory to store all the displacement files
    if os.path.isdir('displacement_files/') == False:
        os.mkdir('displacement_files/')

    # read in image stack
    image_stack = io.imread(ref_file_list[0][:-14] + '_registered.tif')

    # correct the stack shape if there's only one image
    if len(image_stack.shape) == 2:
        temp = np.zeros((1,image_stack.shape[0],image_stack.shape[1]))
        temp[0] = image_stack.copy()
        image_stack = temp.copy()

    # Get the number of images in the stack
    N_images = image_stack.shape[0]

    for t, image_stack_plane in enumerate(image_stack):
        
        # perform optical flow forward and backward
        flow_forward = cv2.calcOpticalFlowFarneback(reference_image, image_stack_plane, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
        flow_reverse = cv2.calcOpticalFlowFarneback(image_stack_plane, reference_image, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, 0)
        
        # take the average of the two flow fields
        flow = (flow_forward - flow_reverse) / 2

        # save the displacements as images
        io.imsave('displacement_files/disp_u_%03d.tif' % (t), flow[:,:,0], check_contrast=False)
        io.imsave('displacement_files/disp_v_%03d.tif' % (t), flow[:,:,1], check_contrast=False)


    return