# -*- coding: utf-8 -*-
"""

Author - Daniel J. Whiting 
Date modified - 10/08/2017
--- Installation ---
Requires standard python distribution including numpy and matplotlib
--- Features ---
A program to calculate the propagation of a 1D electric field
through free space and various optical elements. The program 
uses the angular spectrum method of fourier propagation and currently 
supports propagation through thick plano-convex lenses.
--- Usage ---
The user enters properties of the input field and sets up a sequence of propagation
steps before running the script.
--- Changelog ---

"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def gen_X(dx,xmax):
    ''' Generate a set of symmetric x-axis points '''
    X0 = np.array([0])
    X1 = np.arange(dx,xmax,dx)
    X2 = -X1[::-1]
    X = np.concatenate((X2,X0,X1))
    return X

def gen_freq(nx):
    ''' Calculate the frequency axis based on the number of position points '''
    freq = np.fft.fftshift(np.fft.fftfreq(nx))
    return freq

def f_propagate(E,dz,kz,n0=1):
    ''' Fourier propogation by the angular spectrum method '''
    A_old = np.fft.fftshift(np.fft.fft(E))                  # fourier transform
    A_new = A_old*np.e**(1j*n0*kz*dz)                       # apply propogator
    E = np.fft.ifft(np.fft.ifftshift(A_new))                # inverse transform
    return E

def f_propagate_multi(E,Z,kz,n0=1):
    ''' Fourier propogation by the angular spectrum method '''
    A_old = np.fft.fftshift(np.fft.fft(E))                  # fourier transform
    data = np.zeros((len(E),len(Z)),dtype='complex')
    for i in range(0,len(Z)):
        A_new = A_old*np.e**(1j*n0*kz*Z[i])                 # apply propogator
        data[:,i] = np.fft.ifft(np.fft.ifftshift(A_new))    # inverse transform
    return data

def add_thin_lens_phase(data,f,k,X):
    ''' Add a thin lens phase front to a 1D array '''
    phi = -k*X**2/(2*f) # Thin lens
    return data*np.e**(1j*phi)

def propagate_lens_pconvex(E,X,R,D,t_edge,n,dz,n0=1,orientation='forward'):
    ''' Propagates E through a thick plano-convex lens '''
    t0 = R - (R**2 - (D/2)**2)**.5
    t_centre = t0 + t_edge
    Z = np.arange(0,t_centre+dz,dz)
    for iz in range(len(Z)):
        n_x = n0*np.ones(len(X))
        if orientation == 'forward':
            if Z[iz] < t_edge:
                n_x[abs(X)<D/2] = n
            else:
                n_x[(Z[iz] + R - t_centre)**2 + X**2 < R**2] = n
        elif orientation == 'reverse':
            if Z[iz] > t0:
                n_x[abs(X)<D/2] = n
            else:
                n_x[(Z[iz] - R)**2 + X**2 < R**2] = n
        else:
            ' Invalid orientation: must be "forward" or "reverse". '
        phi = k*n_x*dz
        E = E*np.e**(1j*phi)
        E = f_propagate(E,dz,kz,n0)
    return E

def aperture(data,X,aperture_radius):
    ''' Apply an aperture function to a 1D array '''
    aperture_upper_index = np.where(X>aperture_radius)[0][0]
    aperture_lower_index = len(X)-aperture_upper_index
    data[:aperture_lower_index] = 0
    data[aperture_upper_index:] = 0
    return data

def propagate_4f_thin(E0,nZ,f,X,n0,k,kz,aperture_radius):
    ''' Propagates E0 through the 4f optical system '''
    N = int(nZ/4)
    Z = np.arange(0,4*f,f/N)
    data = np.zeros((len(E0),len(Z)),dtype = 'complex64')
    # Propagate through 4-f system
    data[:,:N+1] = f_propagate_multi(data[:,0],Z[:N+1],n0,kz)
    data[:,N] = add_thin_lens_phase(data[:,N],f,k,X)
    data[:,N] = aperture(data[:,N],X,aperture_radius)
    data[:,N:3*N+1] = f_propagate_multi(data[:,N],Z[N:3*N+1],n0,kz)
    data[:,3*N] = add_thin_lens_phase(data[:,3*N],f,k,X)
    data[:,3*N] = aperture(data[:,3*N],X,aperture_radius)
    data[:,3*N:] = f_propagate_multi(data[:,3*N],Z[3*N:],n0,kz)
    return data,Z

def sinc(X,w0):
	E0 = np.sinc(X/w0)
	return E0

def s_gauss(X,w0,n):
    ''' Super Gaussian: e^[(x/w)^n] '''
    E0 = np.e**(-abs(X/w0)**n)
    return E0

if __name__ == "__main__":

    # -------------- User set parameters -----------------------
    lam = 3e8/550e9             # wavelength of the field (in m)
    dx = lam/10                 # x axis resolution (in m)
    nx = 2**14-1                # number of x axis points

    # ---------- Automatically generated parameters ------------
    xmax = nx*dx/2              # maximum x axis value (in m)
    X = gen_X(dx,xmax)

    freq = gen_freq(nx)         # np.fft.fftfreq(nx)
    k=2*np.pi/lam               # wavenumber
    kx=2*np.pi * freq/dx        # x component of the wavevector
    kx=kx.astype('complex64')
    kz = (k**2-kx**2)**0.5      # z component of the wavevector
    #kz = k - (1/2)*kx**2/k     # paraxial approximation on kz

    # ----------- Set initial electric field -------------------
    w = 2e-3                    # Waist
    g = s_gauss(X,w,2)          # Super Gaussian: e^[(x/w)^n]
    #g = sinc(X,w)
    E0 = g/abs(g).max()         # Normalise the aperture function

    # ------------------- Running ------------------------------
    # lens parameters (if using thick lens functions)
    R = 32.25e-3
    D = 50e-3
    t_edge = 4.6e-3
    n = 1.43
    dz = 0.01e-3

    # propagation of electric field
    E = E0
    E = f_propagate(E,62e-3,kz)
    E = aperture(E,X,D/2)
    #E = add_thin_lens_phase(E,150e-3,k,X)
    E = propagate_lens_pconvex(E,X,R,D,t_edge,n,dz,orientation='forward')
    E = f_propagate(E,250e-3,kz)
    #E = add_thin_lens_phase(E,75e-3,k,X)
    E = aperture(E,X,D/2)
    E = propagate_lens_pconvex(E,X,R,D,t_edge,n,dz,orientation='reverse')
    z = np.linspace(0,85e-3,100)
    E_array = f_propagate_multi(E,z,kz)

    # ------------------- Plotting -----------------------------
    plt.figure()
    plt.title('2D electric field propagation')
    plt.ylabel('x position (m)')
    plt.xlabel('z position (m)')
    plt.imshow(abs(E_array)**2,aspect='auto',cmap='viridis',
                extent=[0,z.max(),X[0],X[-1]],origin='lower left')
    plt.ylim(-10e-3,10e-3)
    plt.colorbar()
    plt.tight_layout()

    plt.figure()
    plt.title('Initial and final intensity profiles')
    plt.xlabel('x position (m)')
    plt.ylabel('Intensity')
    plt.plot(X,abs(E0)**2)
    plt.plot(X,(abs(E_array[:,-1])))
    plt.xlim(-10e-3,10e-3)

    plt.show()