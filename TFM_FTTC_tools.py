
import numpy as np                                             # basic math
from scipy.fft import fft2, ifft2, ifftshift               # FFT for image registration
from scipy.interpolate import griddata, SmoothBivariateSpline  # Interpolation for fixing bad PIV vectors
from scipy import optimize                                     # curve fitting
from scipy.sparse import spdiags, csr_matrix, linalg           # sparse matrix algebra
import matplotlib.pyplot as plt                                # for plotting
import glob as glob                                            # grabbing file names
import skimage.io as io                                        # reading in images
import openpiv.pyprocess
import scipy.sparse as sparse
import scipy.linalg as linalg
from scipy import optimize                                     # curve fitting
try:
    from sksparse.cholmod import cholesky_AAt
except:
    print('scikit-sparse not installed - Cannot use the function calculate_regularization_parameter! It must be entered by hand ')


def calculate_fourier_modes(mesh_size, i_max, j_max, lanczos_exp=0):
    '''
    Calculate the wave vectors based on the mesh size
    '''
    kx_vec = 2. * np.pi / i_max / mesh_size * np.append(np.arange(0, (i_max/2)), np.arange(-i_max/2, 0))
    ky_vec = 2. * np.pi / j_max / mesh_size * np.append(np.arange(0, (j_max/2)), np.arange(-j_max/2, 0))
    kx = np.kron(np.ones((1,j_max)), np.array([kx_vec]).T)
    ky = np.kron(np.ones((i_max,1)), ky_vec)
    # filter if desired
    lanczosx = np.sinc(kx * mesh_size / np.pi)**lanczos_exp
    lanczosy = np.sinc(ky * mesh_size / np.pi)**lanczos_exp
    kx[0, 0] = 1    # otherwise (kx**2 + ky**2)**(-1/2.) will be inf
    ky[0, 0] = 1
    return kx, ky, lanczosx, lanczosy   

def calculate_greens_function(E, s, kx, ky, i_max, j_max, mesh_size, bead_depth):
    '''
    Calculate Greens function in Fourier space
    All units are in pixels^-1
    '''
    V = 2 *(1 + s) / E
    # calculate matrices of kx and ky
    kx_sq = kx**2
    ky_sq = ky**2
    kabs = np.sqrt(kx_sq + ky_sq)
    kabs_sq = kx_sq + ky_sq
    
    z = - bead_depth
    h = s - z * kabs / 2.0
    GFt = V * np.exp(z * kabs) * kabs**(-3) * np.array([[kabs_sq - h * kx_sq,
                                                         - h * kx * ky], 
                                                        [- h * kx * ky,
                                                         kabs_sq - h * ky_sq]])

    GFt[:, :, 0, 0] = 0.0           # we assume that all the sources of traction are in the field of view
    return GFt

def calculate_Ginv(GFt, LL):

    # make the regularization identity matrix
    LL_I = LL * np.ones((GFt.shape[2],GFt.shape[3]))

    # break up the GFt array into it's three unique components
    Gxx = GFt[0,0]
    Gxy = GFt[0,1]
    Gyy = GFt[1,1]
    
    # calculate the determinant of the matrix
    determinant = (LL_I**2 + LL_I * Gxx**2 + 2 * LL_I * Gxy**2 + LL_I * Gyy**2 + Gxx**2 * Gyy**2 - 2 * Gxx * Gxy**2 * Gyy + Gxy**4)**(-1)

    # calculate the three components
    Ginv_xx = determinant * (Gxx * (Gyy**2 + LL_I) - Gxy**2 * Gyy)
    Ginv_xy = determinant * (Gxy**3 + Gxy * (LL_I - Gxx * Gyy))
    Ginv_yy = determinant * (Gyy * (Gxx**2 + LL_I) - Gxx * Gxy**2)

    return Ginv_xx, Ginv_xy, Ginv_yy



def calculate_coefficient_matrix(GFt):
    '''
    Matrix for use in Bayesian approach to determining regularization parameter
    '''
    G1 = GFt[0,0].flatten('F')
    G2 = GFt[1,1].flatten('F')
    X1 = np.vstack((G1,G2)).flatten('F')

    G3 = GFt[0,1].flatten('F')
    G4 = np.zeros_like(G3)
    X2 = np.vstack((G4,G3)).flatten('F')
    X3 = X2[1:]
    X = sparse.spdiags([np.append(X3,0), X1, np.append(0, X3)], [-1,0,1], X1.shape[0], X1.shape[0], format="csr")
    
    return X

def calculate_evidence(alpha, *args):
    '''
    Function to calculate the bayesian evidence for regularization
    '''
    # unpack arguments
    beta, GFt, u, C_a, BX_a, X, fuu, M_rows, M_cols, constant = args
    # beta, u, C_a, BX_a, X, fuu, M_rows, M_cols, constant, kx, ky, E, s, meshsize = args
    
    # regularization parameter
    LL = alpha/beta

    # Calculate the inverse Green's function 
    G_inv_xx, G_inv_xy, G_inv_yy = calculate_Ginv(GFt, LL)
    # G_inv_xx, G_inv_xy, G_inv_yy = calculate_Ginv(kx, ky, E, s, u.shape[1], u.shape[2], meshsize, LL)

    # Perform the actual inverse problem
    Ftfx, Ftfy = reg_fourier_TFM_L2(u, G_inv_xx, G_inv_xy, G_inv_yy)

    # reshape the forces into a single row
    fxx = Ftfx.flatten('F')
    fyy = Ftfy.flatten('F')
    f = np.vstack((fxx,fyy)).flatten('F')

    # construct coefficient matrix for bayesian evidence
    A = alpha * C_a + BX_a
    L = cholesky_AAt(A.T) 
    logdetA = L.logdet() / 2
    Xf_u = X * f - fuu

    # Recovered displacments
    Ftux1= Xf_u[::2]
    Ftuy1= Xf_u[1::2]

    ff = np.sum(np.sum(Ftfx*np.conj(Ftfx) + Ftfy*np.conj(Ftfy))) / (0.5*M_cols)
    uu = np.sum(np.sum(Ftux1*np.conj(Ftux1) + Ftuy1*np.conj(Ftuy1))) / (0.5*M_rows)
    
    # calculate the bayesian evidence
    evidence_value = -0.5 * (-alpha * ff - beta * uu - logdetA + M_cols * np.log(alpha) + constant)
    
    return evidence_value

def gaussian_fit(intensity, A, B, C):
    return A * np.exp(-0.5 * ((intensity - B) / C) ** 2)

def estimate_displacement_variance(u, plot_distribution = False):
    '''
    Estimate the variance of the noise in the displacement field by fitting a gaussian to the major
    peak in the displacement histogram
    '''
    # histogram of displacement vectors
    counts, bin_edges = np.histogram(u.ravel(), bins=200)
    # get bin centers instead of edges
    bins = bin_edges[:-1] + np.diff(bin_edges)/2
    
    # initial parameter guesses
    # A is going to be related to the maximum of the curve
    # B is going to be related to where that maximum is
    # C is going to be related to the width of that curve
    hist_max = np.argwhere(counts == np.max(counts))[0][0]
    p0 = [np.max(counts), bins[hist_max], np.std(u.ravel())]
    
    # find the points you want to include in the histogram 
    counts_mask = np.argwhere(counts > counts[hist_max]/2)

    # Fit the curve
    params, params_covariance = optimize.curve_fit(gaussian_fit, bins[counts_mask[0][0]:counts_mask[-1][0]], counts[counts_mask[0][0]:counts_mask[-1][0]], p0)

    # variance is the square of the standard deviation
    variance = params[2]**2
    
    if plot_distribution:
        # Create a fit line using the parameters from your fit and the original bins
        bg_fit = gaussian_fit(bins, params[0], params[1], params[2])

        # Display the plot with the fit on top of it
        bg_fit_fig, bg_fit_axes = plt.subplots()
        bg_fit_axes.scatter(bins,counts, c='k', label='Image')
        bg_fit_axes.plot(bins, bg_fit, c='r', label='BG Fit')
        bg_fit_axes.legend(loc='best')
        bg_fit_axes.set_xlabel('Displacement')
        bg_fit_axes.set_ylabel('Counts')
        bg_fit_fig.show()
        
    return variance

def calculate_regularization_parameter(GFt, u, beta, kx, ky, E, s, meshsize):
    '''
    This is taken straight from:
    Yunfei Huang, Gerhard Compper, Benedikt Sabass
    A Bayesian tractin force microscopy method with automated denoising in a user-friendly software package
    Computer Physics Communications 256 (2020) 107313
    '''

    # generate sparse matrix
    X = calculate_coefficient_matrix(GFt)
    
    # Fourier transform the displacements
    ux_fft = fft2(u[0]).flatten('F')
    uy_fft = fft2(u[1]).flatten('F')
    u_fft = np.vstack((ux_fft, uy_fft)).flatten('F')
    
    ## Matrices and values needed for BFTTC
    M_rows, M_cols = X.shape
    C = sparse.identity(M_rows).tocsr()
    XX = np.dot(X.T, X)
    BX_a = beta * XX / M_rows * 2
    C_a = C / M_cols * 2
    constant = M_rows * np.log(beta) - M_rows * np.log(2 * np.pi)
    
    # set min and max potential alphas
    alpha_min =1e-12 
    alpha_max =1e2
    
    # calculate optimal alpha by minimizing evidence
    alpha_opt = optimize.golden(calculate_evidence, brack=(alpha_min, alpha_max), args=(beta, GFt, u, C_a, BX_a, X, u_fft, M_rows, M_cols, constant))

    # regularization parameter is 
    LL = np.real(alpha_opt)/beta
    
    return LL

def reg_fourier_TFM_L2(u, Ginv_xx, Ginv_xy, Ginv_yy):
    '''
    Calculate the Fourier Transformed Forces using Ginv
    '''
    Ftux = np.fft.fft2(u[0])
    Ftuy = np.fft.fft2(u[1])
    Ftfx = Ginv_xx * Ftux + Ginv_xy * Ftuy
    Ftfy = Ginv_xy * Ftux + Ginv_yy * Ftuy
    return Ftfx, Ftfy

def reconstruct_displacement_field(GFt, Ftfx, Ftfy, lanczosx, lanczosy):
    # use the Green's function and calculated Forces to predict displacements
    Ftux_rec = GFt[0, 0] * Ftfx + GFt[0, 1] * Ftfy
    Ftuy_rec = GFt[1, 0] * Ftfx + GFt[1, 1] * Ftfy
    # Fourier transform displacements back into real space
    ux_rec = np.fft.ifft2(lanczosx * Ftux_rec)
    uy_rec = np.fft.ifft2(lanczosy * Ftuy_rec)
    # reshape into a single matrix
    urec = np.array([np.real(ux_rec), np.real(uy_rec)])
    Fturec = np.array([Ftux_rec, Ftuy_rec])
    return urec, Fturec

def calculate_stress_field(Ftfx, Ftfy, lanczosx, lanczosy, grid_mat, u, i_max, j_max, \
                           i_bound_size, j_bound_size, pix_per_mu, mesh_size):
    fx = np.fft.ifft2(lanczosx * Ftfx) 
    fy = np.fft.ifft2(lanczosy * Ftfy)
    pos = np.array([np.reshape(grid_mat[0], (i_max*j_max)), 
                        np.reshape(grid_mat[1], (i_max*j_max))])
    vec = np.array([np.reshape(u[0], (i_max*j_max)),
                        np.reshape(u[1], (i_max*j_max))])
    f = np.array([np.real(fx), np.real(fy)])
    fnorm = (f[0]**2 + f[1]**2)**0.5
   
    energy = calculate_energy(u, f, pix_per_mu, mesh_size)
    return pos, vec, fnorm, f, energy

def calculate_energy(u, f, pix_per_mu, mesh_size):
    l = mesh_size / pix_per_mu * 1e-6   # nodal distance in the rectangular grid in m**2 -> dA = l**2
    energy = 0.5 * l**2 * np.sum(u * f) * 1e-6 / pix_per_mu # u is given in pix -> additional 1e-6 / pix_per_mu, f is given in Pa
    return energy

def find_regularization_parameter(shear_modulus = 16000):
    # calculate the best regularization parameter
    ux = io.imread('displacement_files/disp_u_001.tif')
    uy = io.imread('displacement_files/disp_v_001.tif')
    u = np.zeros((2,ux.shape[0],ux.shape[1]))
    u[0,:,:] = ux
    u[1,:,:] = uy

    _, i_max, j_max = u.shape

    # poisson ratio
    nu = 0.5
    meshsize = 1
    # convert shear modulus to Young's modulus
    E = shear_modulus* 3
    bead_depth = 0

    variance = estimate_displacement_variance(u)
    beta = 1 / variance

    kx, ky, lanczosx, lanczosy = calculate_fourier_modes(meshsize, i_max, j_max, lanczos_exp=0)

    GFt = calculate_greens_function(E, nu, kx, ky, i_max, j_max, meshsize, bead_depth)
    LL = calculate_regularization_parameter(GFt, u, beta, kx, ky, E, nu, meshsize)

    return LL
