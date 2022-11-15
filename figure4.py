# -*- coding: utf-8 -*-
"""
Created on Wed May 11 12:45:29 2022

@author: Paul

Plotting histograms of kendall tau of EWS & statistics of tas for cVeg TP grid points
and first 40 years for all grid points
"""

import iris
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import scipy.signal
import seaborn as sns
import statsmodels.tsa.seasonal
import statsmodels.tsa.stattools
import tqdm
from iris.cube import Cube
from scipy import stats
from scipy.io import loadmat

pcolors = ["blue", "red"]
linestyles = ["solid", "dashed", "dotted"]

plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["font.family"] = "serif"

################# Specify variable of interest ######################
var = 'cVeg'  #'treeFrac'

# Specify experiments
experiment = '1pctCO2'
experiment2 = 'piControl'

####### Specify name of directory of data which you want to run algorithm on ############

############ Specify name of directory of that stores analysis data #################
region2 = 'Amazon'  #'World'#

# Window length used to calculate abrupt shift over
wl = 15
EWS_wl = 50  #40
tau_years_end = 5  #10#10#10
tau_years_start = 25  #20 #20#30#tau_years_end+10#45 #25
tau_years_len = tau_years_start - tau_years_end

minlen = 70

co2_start = 284.3186666723341
co2 = np.zeros(165)
for k in range(165):
    co2[k] = co2_start * (1.01**k)

wl_step = 1
string = '_wlstep_' + str(wl_step) + 'months'  #''

################# Specify models here ##################################
models = ['EC-Earth3-Veg', 'GFDL-ESM4', 'NorCPM1', 'SAM0-UNICON', 'TaiESM1']
variant_ids = ['r1i1p1f1', 'r1i1p1f1', 'r1i1p1f1', 'r1i1p1f1', 'r1i1p1f1']
date_ends = ['2000', '1999', '2013', '1999', '1999']
# models = ['EC-Earth3-Veg','GFDL-ESM4','NorCPM1','SAM0-UNICON','TaiESM1']
# variant_ids = ['r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f1','r1i1p1f1']
# date_ends = ['2000','1999','2013','1999','1999']

################## Specify path to analysis data directory #####################
path = path2 = path3 = "./"

#path = '../CMIP6_data/'+var+'/'+experiment+'/Analysis_data/'+region2+'/'
#path2 = '../CMIP6_data/'+var+'/'+experiment2+'/Analysis_data/'+region2+'/'
#path3 = '../CMIP6_data/'+var+'/'+experiment+'/Processed_data/'+region2+'/'

wl_step = 1
string = '_wlstep_' + str(wl_step) + 'months'  #''

Tip_ind = []
Tip_ind_long_enough = []
co2_times = []
nx = 49
ny = 34

AR1_matrix = np.zeros((len(models), int((150 - EWS_wl) * 12 - 1), ny, nx))
Var_matrix = np.zeros((len(models), int((150 - EWS_wl) * 12 - 1), ny, nx))
lam_matrix = np.zeros((len(models), int(150 * 12), ny, nx))

AR1_tau = np.zeros((len(models), ny, nx))
Var_tau = np.zeros((len(models), ny, nx))
lam_tau = np.zeros((len(models), ny, nx))

for i in range(len(models)):

    ################## Specify filename of saved analysis data and load in #####################
    mat = loadmat(path + var + '_' + models[i] + '_as_grads_wl' + str(wl) +
                  '_data_' + experiment + '_' + region2 + '.mat')
    mat2 = loadmat(path2 + var + '_' + models[i] + '_wl' + str(wl) + '_data_' +
                   experiment2 + '_' + region2 + '.mat')
    #mat3 = loadmat(path+var+'_lag_1month_TP_EWS_'+models[i]+'_'+experiment+'_'+region2+'_NSA_v3.mat')
    #mat4 = loadmat(path+var+'_TP_spectral_EWS_'+models[i]+'_'+experiment+'_'+region2+'_NSA_v3.mat')

    cube = iris.load_cube(
        path3 + var + '_' + models[i] + '_' + experiment + '_' +
        variant_ids[i] + '_1850-' + date_ends[i] + '_' + region2 +
        '_monthly.nc', var)

    nt = len(cube.data[:, 0, 0])
    data = np.zeros((nt - 1, ny, nx))
    if models[i] == 'TaiESM1':
        nt = nt - 1
    for j in range(nt - 1):
        data[j, :, :] = cube.data[j + 1, :, :] - cube.data[j, :, :]
    # if models[i] == 'GFDL-ESM4':
    #     stop

    # Extract data from dictionary (only extract Amazon points)
    Lon, Lat, as_grad, as_grads, co2_tip, as_change, ovr_change, control_grads = mat[
        'Lon'][36:, :], mat['Lat'][36:, :], mat['as_grad'][
            36:, :], mat['as_grads'][36:, :], mat['co2_tip'][
                36:, :], mat['as_change'][36:, :], mat['ovr_change'][
                    36:, :], mat2['control_grads'][:, 36:, :]
    #AR1_matrix[i,:,:,:], Var_matrix[i,:,:,:] = mat3['Aut_mat'][:], mat3['Var_mat'][:]
    #lam_matrix[i,:,:,:] = mat4['lam_mat'][:]

    if i == 0:
        Lon_flat, Lat_flat = Lon.flatten(), Lat.flatten()
        points = np.vstack((Lon_flat, Lat_flat)).T

        NSA_verts = [(-72, 12), (-72, -8), (-50, -8), (-50, 7.6), (-55, 12)]

        NSA_path = mpltPath.Path(NSA_verts)
        NSA_grid = NSA_path.contains_points(points)
        NSA_grid = NSA_grid.reshape((ny, nx))
        NSA_mask = np.broadcast_to(NSA_grid, (len(models), ny, nx))

    control_grad_mean = np.nanmean(control_grads[-400:-wl, :, :], axis=0)
    control_grad_std = np.nanstd(control_grads[-400:-wl, :, :], axis=0)

    ############ Set minimum number of standard deviations from ############
    ############ zero for it to be classed as an abrupt shift ############
    std_threshold = 3

    ############ Set minimum threshold for change in sliding window to #############
    ############ be classed as an abrupt shift #####################################
    abs_change_threshold = 2  #10#

    as_frac_threshold = 0.25

    # indxs = ((co2_tip>=co2[years])&(np.abs(as_grad)>std_threshold*control_grad_std)&(np.abs(as_change)>abs_change_threshold)&(np.abs(as_change/ovr_change)>as_frac_threshold))
    indxs = ((np.abs(as_grad) > std_threshold * control_grad_std) &
             (np.abs(as_change) > abs_change_threshold) &
             (np.abs(as_change / ovr_change) > as_frac_threshold))
    indxs2 = ((ovr_change < 0) & (as_change < 0) & (indxs) & (NSA_grid))

    # nn = 0
    # mm = 0

    for k in range(nx):
        for j in range(ny):

            TP_month = 12 * np.argmin(np.abs(co2_tip[j, k] - co2))

            if (indxs2[j, k]) and (TP_month > 12 * (tau_years_start + EWS_wl)):
                # if models[i] == 'SAM0-UNICON':
                #     stop
                AR1_tau[i, j, k], _ = stats.kendalltau(
                    np.arange(12 * tau_years_len),
                    AR1_matrix[i, TP_month - 12 * (tau_years_start + EWS_wl) -
                               1:TP_month - 12 * (tau_years_end + EWS_wl) - 1,
                               j, k])
                Var_tau[i, j, k], _ = stats.kendalltau(
                    np.arange(12 * tau_years_len),
                    Var_matrix[i, TP_month - 12 * (tau_years_start + EWS_wl) -
                               1:TP_month - 12 * (tau_years_end + EWS_wl) - 1,
                               j, k])
                lam_tau[i, j, k], _ = stats.kendalltau(
                    np.arange(12 * tau_years_len),
                    -lam_matrix[i, TP_month - 12 * tau_years_start:TP_month -
                                12 * tau_years_end, j, k])
            else:
                AR1_tau[i, j,
                        k], Var_tau[i, j,
                                    k], lam_tau[i, j,
                                                k] = np.nan, np.nan, np.nan

    Tip_ind.append(indxs2)
    Tip_ind_long_enough.append(
        np.logical_and(indxs2, co2_tip > co2[tau_years_start + EWS_wl]))
    co2_times.append(co2_tip)

# stop
Tip_indxs = np.stack(Tip_ind)
Tip_allowed = np.stack(Tip_ind_long_enough)
tipping_time_co2 = np.stack(co2_times)
tipping_index = (np.log(tipping_time_co2 / co2_start) /
                 np.log(1.01)).astype(int) * 12

var_warning = {}
ac_warning = {}
ls_warning = {}
window_len = EWS_wl  #50
for i in range(len(models)):
    print(f"On model {models[i]}")
    cVegcube = iris.load_cube(
        path3 + "cVeg" + '_' + models[i] + '_' + experiment + '_' +
        variant_ids[i] + '_1850-' + date_ends[i] + '_' + region2 +
        '_monthly.nc', "cVeg")
    tascube = iris.load_cube(
        path3 + "tas" + '_' + models[i] + '_' + experiment + '_' +
        variant_ids[i] + '_1850-' + date_ends[i] + '_' + region2 +
        '_monthly.nc', "tas")
    cveg_ts = cVegcube.data[:, Tip_indxs[i]].data
    tas_ts = tascube.data[:, Tip_indxs[i]].data
    cv_det = np.copy(cveg_ts)
    tas_det = np.copy(tas_ts)
    for j in tqdm.trange(cv_det.shape[1], desc="detrend"):
        cv_det[:, j] = statsmodels.tsa.seasonal.STL(cveg_ts[:, j],
                                                    period=12).fit().resid
        tas_det[:, j] = statsmodels.tsa.seasonal.STL(tas_ts[:, j],
                                                     period=12).fit().resid

    vw = np.full_like(cv_det, np.nan)
    ac = vw.copy()
    ls = vw.copy()

    for t in tqdm.trange(window_len * 12, cveg_ts.shape[0],
                         desc="Classic EWS"):
        vw[t] = cv_det[t - window_len * 12:t].var(axis=0)
        for j in range(ac.shape[1]):
            ac[t,
               j] = statsmodels.tsa.stattools.acf(cv_det[t - window_len * 12:t,
                                                         j],
                                                  alpha=0.05,
                                                  fft=True)[0][1]
    var_warning[models[i]] = vw
    ac_warning[models[i]] = ac
    for t in tqdm.trange(window_len * 12,
                         cveg_ts.shape[0],
                         desc="Spectral EWS"):
        f, Sxx = scipy.signal.welch(cv_det[t - window_len * 12:t],
                                    fs=12.0,
                                    detrend=False,
                                    nperseg=window_len * 12,
                                    axis=0)
        f, Sff = scipy.signal.welch(tas_det[t - window_len * 12:t],
                                    fs=12.0,
                                    detrend=False,
                                    nperseg=window_len * 12,
                                    axis=0)
        for j in range(ls.shape[1]):

            def ff(fs, ls):
                return Sff[1:, j] * (1 / 12.0)**2 / (
                    (2 * np.pi * fs)**2 + ls**2)

            p0 = 5.0
            popt, pcov = scipy.optimize.curve_fit(ff,
                                                  f[1:],
                                                  Sxx[1:, j],
                                                  p0=[p0],
                                                  bounds=(0.0, np.inf))
            ls[t, j] = -popt[0]
    ls_warning[models[i]] = ls

indicators = {}
for m in models:
    indicators[m] = {}
for i in range(len(models)):
    print(models[i])
    allowed = np.where(Tip_indxs[i].flatten())[0].searchsorted(
        np.where(Tip_allowed[i].flatten())[0])

    tt = tipping_index[i][Tip_allowed[i]]
    taus = np.zeros(tt.size)
    for j in tqdm.trange(taus.size):
        kt = scipy.stats.kendalltau(
            np.arange(tau_years_len * 12),
            var_warning[models[i]][:, allowed][tt[j] -
                                               tau_years_start * 12:tt[j] -
                                               tau_years_end * 12, j])
        taus[j] = kt[0]
    indicators[models[i]]["var"] = taus.copy()
    for j in tqdm.trange(taus.size):
        kt = scipy.stats.kendalltau(
            np.arange(tau_years_len * 12),
            ac_warning[models[i]][:,
                                  allowed][tt[j] - tau_years_start * 12:tt[j] -
                                           tau_years_end * 12, j])
        taus[j] = kt[0]
    indicators[models[i]]["ac"] = taus.copy()
    for j in tqdm.trange(taus.size):
        kt = scipy.stats.kendalltau(
            np.arange(tau_years_len * 12),
            ls_warning[models[i]][:,
                                  allowed][tt[j] - tau_years_start * 12:tt[j] -
                                           tau_years_end * 12, j])
        taus[j] = kt[0]
    indicators[models[i]]["ls"] = taus.copy()

threshold = np.linspace(1.0, 0.0)
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))
axs = axs.flatten()
for i in range(len(models)):
    idx = i + 1
    varth = np.fromiter((np.sum(
        indicators[models[i]]["var"][indicators[models[i]]["ac"] > 0.0] > th) /
                         indicators[models[i]]["var"].size
                         for th in threshold), "double")
    acth = np.fromiter((np.sum(
        indicators[models[i]]["ac"][indicators[models[i]]["var"] > 0.0] > th) /
                        indicators[models[i]]["ac"].size
                        for th in threshold), "double")
    lsth = np.fromiter((np.sum(indicators[models[i]]["ls"] > th) /
                        indicators[models[i]]["ls"].size
                        for th in threshold), "double")
    axs[idx].plot(threshold, varth, color="blue")
    axs[idx].plot(threshold, acth, color="red")
    axs[idx].plot(threshold, lsth, color="black")
    axs[idx].set_title(models[i])
vh = np.zeros_like(threshold)
ah = np.zeros_like(threshold)
lh = np.zeros_like(threshold)
count = 0.0
for i in range(len(models)):
    vh += np.fromiter((np.sum(
        indicators[models[i]]["var"][indicators[models[i]]["ac"] > 0.0] > th)
                       for th in threshold), "double")
    ah += np.fromiter((np.sum(
        indicators[models[i]]["ac"][indicators[models[i]]["var"] > 0.0] > th)
                       for th in threshold), "double")
    lh += np.fromiter((np.sum(indicators[models[i]]["ls"] > th)
                       for th in threshold), "double")
    count += indicators[models[i]]["var"].size
vh /= count
ah /= count
lh /= count
axs[0].plot(threshold, vh, color="blue")
axs[0].plot(threshold, ah, color="red")
axs[0].plot(threshold, lh, color="black")
axs[0].set_title("All Models")
for ax in axs:
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
fig.supxlabel(r"Kendall $\tau$ Threshold")
fig.supylabel("Proportion of Abrupt Shifts Detected")
plt.tight_layout()
plt.savefig("figure4.eps")
plt.show()
