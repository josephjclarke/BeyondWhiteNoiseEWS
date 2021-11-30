#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:28:47 2021

@author: jc1147
"""

from plotstyles import *
import integrator
import numpy as np
import ews
import scipy.stats

epsilon = 0.01
times = np.linspace(0.0, 1.0, 10000) / epsilon

print("Integrating White Noise")
forcing_white, response_white, noise_white = integrator.integrate(times, np.sqrt(3), "white", epsilon)
print("Integrating Red Noise")
forcing_red, response_red, noise_red = integrator.integrate(times, np.sqrt(3), "red",epsilon)

fs = 1/np.diff(times)[0]
print("Calculating lambdas white")
white_lambdas = ews.lambda_from_spectrum(forcing_white, response_white,2500,fs)**.5
white_fit = scipy.stats.linregress(epsilon*times[2500:6666],white_lambdas[2500:6666])
print("Calculating lambdas red")
red_lambdas = ews.lambda_from_spectrum(forcing_red, response_red,2500,fs)**.5
red_fit = scipy.stats.linregress(epsilon*times[2500:6666],red_lambdas[2500:6666])
  
print("Plotting")
fig, ax = plt.subplots(2,2)
ax[0,0].plot(epsilon*times,response_white,color=pcolors[0])
ax[0,1].plot(epsilon*times,response_red,color=pcolors[0])

ax[1,0].plot(epsilon*times,white_lambdas,color=pcolors[0])
ax[1,1].plot(epsilon*times,red_lambdas,color=pcolors[0])

ax[1,0].plot(epsilon*times[2500:6666],epsilon*times[2500:6666]*white_fit.slope + white_fit.intercept,color=pcolors[1])
ax[1,1].plot(epsilon*times[2500:6666],epsilon*times[2500:6666]*red_fit.slope + red_fit.intercept,color=pcolors[1])

ax[0,0].set_ylabel(r"$x$")
ax[1,0].set_ylabel(r"$|\lambda|$")
ax[1,0].set_xlabel(r"$\epsilon t$")
ax[1,1].set_xlabel(r"$\epsilon t$")

for n,a in enumerate(ax.flatten()):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.text(-0.1, 1.1, string.ascii_uppercase[n], transform=a.transAxes,size=12, weight='bold')
    a.set_xlim(0.0,1.0)
    
plt.tight_layout()
plt.savefig("figure2.eps")
plt.close()
