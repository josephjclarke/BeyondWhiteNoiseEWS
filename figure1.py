#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:28:30 2021

@author: jc1147
"""

from plotstyles import *

import integrator
import numpy as np
import ews

epsilon = 0.01
times = np.linspace(0.0, 1.0, 10000) / epsilon

print("Integrating White Noise")
forcing_white, response_white, noise_white = integrator.integrate(times, np.sqrt(3), "white", epsilon)
print("Integrating Red Noise")
forcing_red, response_red, noise_red = integrator.integrate(times, np.sqrt(3), "red",epsilon)

print("Calculating traditional Early Warning Signals")
vr,ar = ews.traditional_ews(response_red,2500)
vw,aw = ews.traditional_ews(response_white,2500)

print("Plotting")
fig, ax = plt.subplots(2,1)
vax = ax[1].twinx()
ax[0].plot(epsilon*times,response_white,color=pcolors[0])
ax[0].plot(epsilon*times,response_red,color=pcolors[1])
#ax[0].axvline(2.0/3.0,linestyle=linestyles[2])

ax[1].plot(epsilon*times[:6666],aw[:6666]-aw[2500],color=pcolors[0])
ax[1].plot(epsilon*times[:6666],ar[:6666]-ar[2500],color=pcolors[1])
vax.plot(epsilon*times[:6666],vw[:6666]/vw[2500],color=pcolors[0],linestyle=linestyles[1])
vax.plot(epsilon*times[:6666],vr[:6666]/vr[2500],color=pcolors[1],linestyle=linestyles[1])


ax[0].set_xlabel("Time")
ax[0].set_ylabel("State Variable")

ax[1].set_xlabel("Time of End of Window")
ax[1].set_ylabel("Change in Autocorrelation")
vax.set_ylabel("Normalised Variance")

ax[0].set_xlim(0.0,1.0)
ax[1].set_xlim(0.0,1.0)
vax.set_yscale("log")
vax.set_yticks([1.0,10.0])

ax[0].spines["top"].set_visible(False)
ax[0].spines["right"].set_visible(False)

ax[1].spines["top"].set_visible(False)
vax.spines["top"].set_visible(False)

for n,a in enumerate(ax):
    a.text(-0.1, 1.1, string.ascii_uppercase[n], transform=a.transAxes,size=12, weight='bold')
    

plt.tight_layout()
plt.savefig("figure1.eps")
plt.close()