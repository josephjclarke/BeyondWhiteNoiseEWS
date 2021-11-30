# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 13:31:18 2021

@author: Paul

Using power spectrum to determine lambda as an EWS for vegetation carbon at a
particular grid point 
"""

import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
from scipy.stats import kendalltau
import scipy
from scipy import signal
import scipy.optimize as opt
import seaborn as sns
import statsmodels.tsa.tsatools
import statsmodels.nonparametric.smoothers_lowess

def parameters_from_spectrum(forcing, response, dt, x0=[1.0, 1.0], LOGLS=True):
    """Extract the sensitiivity and timescale from a model of the form
    Cdx/dt = F(x,mu(t)) (1)
    which has the linearisation
    Cdy/dt = Ly + f(t) (2)

    Parameters
    ----------
    forcing : array (N,)
        Array of f(t) evaluated at each t
    response: array (N,)
        Array of x(t), evaluated at each t
    dt : float
        the timestep
    x0 : optional 2 element list of floats
        inital guess for L and C, defaults to 1.0 and 1.0
    LOGLS : optional boolean value
        apply log based (True), or standard (False) least squares minimisation

    Returns 
    -------
    L**2,C : float,float
    the sensitiivity squared and the inverse timescale

    Notes:
    Taking the FT of (2) gives
    iwCy(w) = Ly(w) + f(w) (3)
    which can be rearranged to give

    |y(w)|^2 = |f(w)|^2 / (C**2  w**2 + L**2) (4),
    so we can fit (4) to get L**2 and C
    """
    #calculate the (angular) sampling frequency
    fs = 2 * np.pi / dt

    #calculate the spectral density of the forcing
    f, Pxx_f = scipy.signal.periodogram(forcing,
                                        fs,
                                        detrend=False,
                                        scaling="spectrum")

    #calculate the spectral density of the response
    f, Pxx_r = scipy.signal.periodogram(response,
                                        fs,
                                        detrend="constant",
                                        scaling="spectrum")

    #The residual, to minimized in a least squares fitting to the data
    #we ignore the awkward zero frequency contribution
    def residual(x):
        l = x[0]
        C = x[1]
        if LOGLS:
            return np.log10(Pxx_r[1:-1]) - np.log10(Pxx_f[1:-1]) + np.log10((C**2 * f[1:-1]**2 + l**2))
        else:
            return Pxx_r[1:] - Pxx_f[1:] / (C**2 * f[1:]**2 + l**2)
        

    #the jacobian used in the fitting
    def residual_jac(x):
        l = x[0]
        C = x[1]
        if LOGLS:
            jac_l = 2 * l / (C**2 * f[1:-1]**2 + l**2) / np.log(10)
            jac_C = 2 * C * f[1:-1]**2  / (C**2 * f[1:-1]**2 + l**2) / np.log(10)
        else:
            jac_l = 2 * l * Pxx_f[1:] / (C**2 * f[1:]**2 + l**2)**2
            jac_C = 2 * C * f[1:]**2 * Pxx_f[1:] / (C**2 * f[1:]**2 + l**2)**2


        return np.asarray([jac_l, jac_C]).T

    #calculate the optimal values for L and C to minimize the squared summed residuals
    optimal = scipy.optimize.least_squares(fun=residual,
                                           x0=x0,
                                           jac=residual_jac)

    l = optimal.x[0]
    C = optimal.x[1]

    return l**2, C


def parameters_from_spectrum_rolling(forcing,
                                     response,
                                     dt,
                                     window,
                                     x0=[1.0, 1.0],
                                     LOGLS=True):
    """Extract the sensitiivity and timescale from a model of the form
    Cdx/dt = F(x,mu(t)) (1)
    which has the linearisation
    Cdy/dt = Ly + f(t) (2)
    in rolling windows

    Parameters
    ----------
    forcing : array (N,)
        Array of mu(t) evaluated at each t
    response: array (N,)
        Array of x(t), evaluated at each t
    dt : float
        the timestep
    window : int
        the number of elements in each moving window. Must be < N
    x0 : optional 2 element list of floats
        inital guess for L and C, defaults to 1.0 and 1.0
    LOGLS : optional boolean value
        apply log based (True), or standard (False) least squares minimisation

    Returns 
    -------
    L2,C : float array(N,), float array(N,)
    the sensitiivity squared and the inverse timescale, elements in [0:window] 
    are set to nan. 
    """

    #arrays to output
    L2 = np.full_like(response, np.nan)
    C = np.full_like(response, np.nan)

    #for each window...
    for i in range(L2.size - window):
        #calculate L**2 and C:
        L2[window + i], C[window + i] = parameters_from_spectrum(
            forcing[i:window + i], response[i:window + i], dt, x0, LOGLS)
    return L2, C


# You shouldn't need to change the below entries at least for the 3 models you
# are currently using
experiment = '1pctCO2'


################# Specify variables of interest ######################
# vars2 is the same as var1 but all lower case
var1 = 'cVeg'
var2 = 'cveg'
var3 = 'tas'

############# List of models to consider #############################
models = ['TaiESM1','UKESM1-0-LL'] 
variant_ids = ['r1i1p1f1','r1i1p1f2']
date_ends = ['1999','1999']

# Region of interest
region = 'Amazon'

# Is variable measured in per second?     
PERSEC = False

############ Specify longitude and latitude coordinate of interest ##############
lon_0 = -58#-57
lat_0 = 0#-6

# sliding window length (months*years) 
wl = 12*30
X = [i%12 for i in range(0, wl)]
   
################## Specify path to processed data directory #####################
path = './' #'../CMIP6_data/'+var1+'/'+experiment+'/Processed_data/'+region+'/'
path2 = './' #'../CMIP6_data/'+var3+'/'+experiment+'/Processed_data/'+region+'/'

##################### Initialise plotting ################################
plt.rcParams["font.family"] = "serif"
fig, ax = plt.subplots(3,1,figsize=(10,9.5),sharex=True)
ax1 = ax[1].twinx()
colours = ['b','r']


for j in range(len(models)):

    # File name of interpolated data
    fname = path+var1+'_'+models[j]+'_'+experiment+'_'+variant_ids[j]+'_1850-'+date_ends[j]+'_'+region+'_monthly.nc'
    fname2 = path2+var3+'_'+models[j]+'_'+experiment+'_'+variant_ids[j]+'_1850-'+date_ends[j]+'_'+region+'_monthly.nc'
    
    # Extract data
    f = nc.Dataset(fname,'r')
    f2 = nc.Dataset(fname2,'r')
    
    # Extract longitudes and latitudes
    longitude = f.variables['lon'][:]
    latitude = f.variables['lat'][:]
    
    # Extract variable data and convert to yearly units if applicable
    if PERSEC:
        x = f.variables[var2][:150*12,:,:]*3600*24*360
    else:
        x = f.variables[var2][:150*12,:,:]
    
    x2 = f2.variables[var3][:150*12,:,:]
    
    # Close dataset
    f.close()
    f2.close()
    
    # Obtain time dimension size
    nt = int(len(x[:,0,0]))

    # Meshgrid longitude and latitude
    Lon, Lat = np.meshgrid(longitude, latitude)
         
    # Find norm error to centre coordinates of box
    a = abs(Lat-lat_0)+abs(Lon-lon_0)
    
    # Find array indicies of nearest point to coordinates 
    index_j, index_i = np.unravel_index(a.argmin(),a.shape)
    
    # Extra response and forcing data for given latitude and longitude
    x3 = x[:,index_j,index_i].copy()
    x4 = x2[:,index_j,index_i].copy()
    
    Var = np.zeros(nt-wl)
    Aut = np.zeros(nt-wl)
    
    # y = np.array([np.mean(x3[i:i+12]) for i in range(x3.size)]) #deseasonalise
    # z = y-statsmodels.nonparametric.smoothers_lowess.lowess(y,np.arange(y.size),return_sorted=False,frac=0.02) #subract smoothed version of series to get residuals. "Nonlinear detrend".

    for i in range(wl,len(x3)):
        # x3_old = z[i-wl:i].copy()
        # x3_new = z[i-wl+1:i+1].copy()
        x3_oldcopy = x3[i-wl:i].copy()
        x3_newcopy = x3[i-wl+1:i+1].copy()
        for l in range(12):
            x3_oldmean = np.nanmean(x3_oldcopy[np.where(np.asarray(X)==l)])
            x3_oldcopy[np.where(np.asarray(X)==l)] = x3_oldcopy[np.where(np.asarray(X)==l)] - x3_oldmean
            x3_newmean = np.nanmean(x3_newcopy[np.where(np.asarray(X)==l)])
            x3_newcopy[np.where(np.asarray(X)==l)] = x3_newcopy[np.where(np.asarray(X)==l)] - x3_newmean
        x3_old = statsmodels.tsa.tsatools.detrend(x3_oldcopy,order=1)#signal.detrend(x3_oldcopy)#
        x3_new = statsmodels.tsa.tsatools.detrend(x3_newcopy,order=1)#signal.detrend(x3_newcopy)#
        Var[i-wl] = np.var(x3_new)
        Aut[i-wl] = np.sum(x3_old*x3_new)/(np.sqrt(np.sum(x3_old**2))*np.sqrt(np.sum(x3_new**2)))

    # Extract the sensitiivity squared
    L2,_ = parameters_from_spectrum_rolling(x4,x3,1/12,wl,x0=[1.0, 1.0],LOGLS=True)

    # Plot time series of response
    ax[0].plot(np.linspace(0,150,nt-1),x3[:-1],c=colours[j])
    
    if j == 0:
        ax[1].plot(np.linspace(wl/12,84,nt-66*12-wl),Aut[:int(84*12-wl)]-Aut[0],c=colours[j])
        ax1.plot(np.linspace(wl/12,84,nt-66*12-wl),Var[:int(84*12-wl)]/Var[0],c=colours[j],ls='--')
        # ax[2].plot(np.linspace(0,84,nt-66*12),np.sqrt(L2[:84*12]),c=colours[j])
        # ax[0].plot([84,84],[0,30],'k--')
        # ax[1].plot([84,84],[-0.02,0.02],'k--')
        # ax[2].plot([84,84],[0,30],'k--')
    else:            
        ax[1].plot(np.linspace(wl/12,150,nt-wl),Aut-Aut[0],c=colours[j])
        ax1.plot(np.linspace(wl/12,150,nt-wl),Var/Var[0],c=colours[j],linestyle='--')
        # ax[2].plot(np.linspace(0,150,nt-1),np.sqrt(L2[:-1]),c=colours[j])
    
    # Plot time series of sensitivity
    ax[2].plot(np.linspace(0,150,nt-1),np.sqrt(L2[:-1]),c=colours[j])

# Plot labelling
ax[0].set_ylabel('Vegetation carbon ($kgC/m^2$)',fontsize=14)
ax[1].set_ylabel('Change in autocorrelation',fontsize=14)
ax1.set_ylabel('Normalised variance',fontsize=14)
ax[2].set_xlabel('Time (years)',fontsize=14)
ax[2].set_ylabel('$|\lambda|$',fontsize=14)
ax[0].tick_params(axis='y', labelsize=12)
ax[2].tick_params(axis='both', labelsize=12)
ax[2].set_yscale('log')
ax[1].set_yticks([-0.005,0,0.005])

ax[0].set_xlim(0,150)
ax[0].set_ylim(0,27)
ax[2].set_ylim(0.3,40)
# ax1.set_yscale('log')

ax[0].text(-0.1, 1.1, 'A', transform=ax[0].transAxes,size=12, weight='bold')
ax[1].text(-0.1, 1.1, 'B', transform=ax[1].transAxes,size=12, weight='bold')
ax[2].text(-0.1, 1.1, 'C', transform=ax[2].transAxes,size=12, weight='bold')
    
fig.tight_layout()
sns.despine()
ax1.spines["right"].set_visible(True)
