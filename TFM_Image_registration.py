
# toolboxes required:
import glob as glob                                  		 # for finding files in the folder
import skimage.io as io                              		 # for reading/writing images
import numpy as np                                   		 # for math operations
from scipy.fftpack import fft2, ifft2, ifftshift     		 # for fourier transform operations
from skimage.registration import phase_cross_correlation     # registration function

def flat_field_correct_image(image, flat_field_image, dark_image):
    '''
    Corrects for field flatness and aberrations due to camera/objective/lasers
    '''

    # check if the image is a stack
    if len(image.shape) > 2:
        # make an empty matrix to hold the corrected image
        corrected_image = np.zeros_like(image)
        for index, plane in enumerate(image):
            # correct each image
            corrected_image[index] = (plane - dark_image) / flat_field_image
    else:
        # for single plane images
        corrected_image = (image - dark_image) / flat_field_image

    # correct for negative numbers
    corrected_image[corrected_image < 0] = 0
        
    return corrected_image

def TFM_Image_registration(flatfield_correct = False, image_list = None, flatfield_images = None, darkfield_image = None):

    # Check if images are to be corrected
    if flatfield_correct:
        if flatfield_images is None:
            flatfield_correct = False
            print('No flatfield images provided - images will not be corrected')
        elif len(flatfield_images) != (len(image_list) + 1):
            flatfield_correct = False
            print('Incorrect number of flatfield images provided - images  will not be corrected')

        if darkfield_image is None:
            flatfield_correct = False
            print('No darkfield images provided - images will not be corrected')
    
    # check if reference has already been corrected
    original_reference_image = glob.glob('*reference_original.tif')
    if len(original_reference_image) > 0:
        reference_image = io.imread(original_reference_image[0], plugin='tifffile', is_ome=False)
        # Correct the reference image
        if flatfield_correct:
            reference_image = flat_field_correct_image(reference_image, flatfield_images[0], darkfield_image)

            io.imsave(original_reference_image[0][:-13] + '.tif', reference_image.astype('uint16'), check_contrast=False)
        file_name = original_reference_image[0][:-23]
    else:
        # Otherwise find and read in your reference image
        file_list = glob.glob('*_reference.tif')
        reference_image = io.imread(file_list[0], plugin='tifffile', is_ome=False)
        # Correct the reference image
        if flatfield_correct:
            io.imsave(file_list[0][:-4] + '_original.tif', reference_image, check_contrast=False)
            reference_image = flat_field_correct_image(reference_image, flatfield_images[0], darkfield_image)
            io.imsave(file_list[0],reference_image.astype('uint16'), check_contrast=False)
        file_name = file_list[0][:-14]
    N_rows = reference_image.shape[0]
    N_cols = reference_image.shape[1]

    # read in your bead image
    image_stack = io.imread(file_name + '.tif')

    # correct the stack shape if there's only one image
    if len(image_stack.shape) == 2:
        temp = np.zeros((1,image_stack.shape[0],image_stack.shape[1]))
        temp[0] = image_stack.copy()
        image_stack = temp.copy()

    if flatfield_correct:
        image_stack = flat_field_correct_image(image_stack, flatfield_images[0], darkfield_image)

    # Get the number of images in the stack
    N_images = image_stack.shape[0]

    # Get the reference image intensity
    reference_image_intensity = np.sum(reference_image)

    # Create an empty array to hold the registered images
    image_stack_registered = np.zeros_like(image_stack)

    # Create an empty array to hold the shift coordinates to reference for use in other channels
    shift_coordinates = np.zeros((N_images, 4))

    # Create a matrix with the row and column numbers for the registered image calculation
    Nr = ifftshift(np.arange(-1 * np.fix(N_rows/2), np.ceil(N_rows/2)))
    Nc = ifftshift(np.arange(-1 * np.fix(N_cols/2), np.ceil(N_cols/2)))
    Nc, Nr = np.meshgrid(Nc, Nr)

    # Define the subpixel resolution
    subpixel_resolution = 100

    # For loop to register each plane in the stack
    for plane in np.arange(0,N_images):
        # Read in the image you want to register
        original_image = image_stack[plane,:,:]
        # Perform the subpixel registration
        shift, error, diffphase = phase_cross_correlation(reference_image, original_image, upsample_factor=subpixel_resolution)
        # Store the shift coordinates
        shift_coordinates[plane] = np.array([shift[0], shift[1], error, diffphase])
    
        # Calculate the shifted image
        shifted_image_fft = fft2(original_image) * np.exp(
                1j * 2 * np.pi * (-shift[0] * Nr / N_rows - shift[1] * Nc / N_cols))
        shifted_image_fft = shifted_image_fft * np.exp(1j * diffphase)
        # normalize the image intensity
        shifted_image = np.abs(ifft2(shifted_image_fft)) * reference_image_intensity / np.sum(original_image)
        image_stack_registered[plane,:,:] = shifted_image.copy()
          
    # Check if image size is odd. If so reduce image size to even
    if N_rows % 2 == 1:
        image_stack_registered = image_stack_registered[:,:-1,:]
    if N_cols % 2 == 1:
        image_stack_registered = image_stack_registered[:,:,:-1]

    # Correct reference image size if odd
    if N_rows % 2 == 1:
        # check if the original reference image file has been saved
        ref_original = glob.glob('*reference_original.tif')
        ref_im = glob.glob('*_reference.tif')
        reference_image = io.imread(ref_im[0])
        if len(ref_original) == 0:
            io.imsave(ref_im[0][:-4] + '_original.tif', reference_image, check_contrast=False)
        reference_image = reference_image[:-1,:]
        io.imsave(ref_im[0], reference_image.astype('uint16'), check_contrast=False)
    if N_cols % 2 == 1:
        # check if the original reference image file has been saved
        ref_original = glob.glob('*reference_original.tif')
        ref_im = glob.glob('*_reference.tif')
        reference_image = io.imread(ref_im[0])
        if len(ref_original) == 0:
            io.imsave(ref_im[0][:-4] + '_original.tif', reference_image, check_contrast=False)
        reference_image = reference_image[:,:-1]
        io.imsave(ref_im[0], reference_image.astype('uint16'), check_contrast=False)


    # Write out the registered stack
    io.imsave(file_name + '_registered.tif', image_stack_registered.astype('uint16'), check_contrast=False)

    # Write out the shifted coordinates
    np.savetxt('shiftcoordinates.txt', shift_coordinates, delimiter=' ', newline='\n')

    if image_list is not None:
        for i,image_name in enumerate(image_list):
            if flatfield_correct == True:
                shift_image_stack(image_name, shift_coordinates, flatfield_images[i+1], darkfield_image, correct_odd_imagesize = True)
            else:
                shift_image_stack(image_name, shift_coordinates, correct_odd_imagesize = True)
                      
    return

def shift_image_stack(image_stack_name, shift_coordinates, flatfield_image =  None, darkfield_image = None, correct_odd_imagesize = True):
    """
    Register an image stack based on a previously registered stack of images
    
    Parameters
    ----------
    image_stack_name  :  str         - name of the image stack to be registered
    shift_coordinates :  4xN ndarray - contains N rows of [ row shift, col shift, error, diffphase] 
                                        from skimage.registration.phase_cross_correlation output. N must be the same number
                                        of images in the stack

    Output
    ------
    Saves the registered stack of images with the same name with '_registered' appended
    """
    
    # read in image stack
    image_stack = io.imread(image_stack_name, plugin='tifffile', is_ome=False).astype('int16')
    
    # correct the stack shape if there's only one image
    if len(image_stack.shape) == 2:
        temp = np.zeros((1,image_stack.shape[0],image_stack.shape[1]))
        temp[0] = image_stack.copy()
        image_stack = temp.copy()

    # If flat field correcting the image
    if (flatfield_image is not None) and (darkfield_image is not None):
        image_stack = flat_field_correct_image(image_stack, flatfield_image, darkfield_image)

    # Get the shape of your stack
    N_planes, N_rows, N_cols = image_stack.shape
    
    # Create a matrix with the row and column numbers for the registered image calculation
    Nr = ifftshift(np.arange(-1 * np.fix(N_rows/2), np.ceil(N_rows/2)))
    Nc = ifftshift(np.arange(-1 * np.fix(N_cols/2), np.ceil(N_cols/2)))
    Nc, Nr = np.meshgrid(Nc, Nr)

    # Create an empty array to hold the registered image
    image_registered = np.zeros((N_planes, N_rows, N_cols))

    # register each plane based on the provided coordinates
    for plane in np.arange(0,N_planes):
        raw_image = image_stack[plane]
        shifted_image_fft = fft2(raw_image) * np.exp(
        1j * 2 * np.pi * (-shift_coordinates[plane,0] * Nr / N_rows - shift_coordinates[plane,1] * Nc / N_cols))
        shifted_image_fft = shifted_image_fft * np.exp(1j * shift_coordinates[plane,3])
        shifted_image = np.abs(ifft2(shifted_image_fft))
        image_registered[plane] = shifted_image.copy()

    # new file name
    image_registered_name = image_stack_name[:-4] + '_registered.tif'
    
    # set any pixels that are less than zero equal to zero
    image_registered[image_registered < 0] = 0

    # Check if image size is odd. 
    if correct_odd_imagesize:
        if N_rows % 2 == 1:
            image_registered = image_registered[:,:-1,:]
        if N_cols % 2 == 1:
            image_registered = image_registered[:,:,:-1]

    # Save the registered stack
    io.imsave(image_registered_name, image_registered.astype('uint16'), check_contrast=False)

    return
