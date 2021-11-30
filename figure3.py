#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:28:56 2021

@author: jc1147
"""

from plotstyles import *
import integrator
import numpy as np
import ews
epsilon = 0.01
times = np.linspace(0.0, 1.0, 10000) / epsilon
fs = 1/np.diff(times)[0]

forcing,x = integrator.integrate_changing_variance_no_tipping(
        times,np.sqrt(3),epsilon,np.ones_like(times)+epsilon*times)

v_no,a_no = ews.traditional_ews(x,2500)
lam_no = ews.lambda_from_spectrum(forcing,x,2500,fs)**.5


fig, ax = plt.subplots(3,1)
ax[0].plot(epsilon*times,x,color=pcolors[0])
vax = ax[1].twinx()
ax[1].plot(epsilon*times,a_no,color=pcolors[0])
vax.plot(epsilon*times,v_no,color=pcolors[1])
ax[2].plot(epsilon*times,lam_no,color=pcolors[0])


ax[2].set_xlabel(r"$0.01 t$")
ax[0].set_ylabel("System State")
ax[1].set_ylabel("Autocorrelation")
vax.set_ylabel("Variance")
ax[2].set_ylabel(r"$|\lambda|$")

vax.ticklabel_format(style='sci', scilimits=(0,0))

for n,a in enumerate(ax.flatten()):
    a.spines["top"].set_visible(False)
    a.spines["right"].set_visible(False)
    a.text(-0.1, 1.1, string.ascii_uppercase[n], transform=a.transAxes,size=12, weight='bold')
    a.set_xlim(0.0,1.0)
vax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("figure3.eps")
plt.close()


#print("Calculating Linear")
#forcing,response,noise = integrator.integrate_linear(times, 0.0)
#print("Calculating Lambda")
#fs = np.abs(1/np.diff(times)[0])
#lambdas = ews.lambda_from_spectrum(forcing, response,2500,fs)**.5
#
#fig, ax = plt.subplots(1,2)
#
#ax[0].plot(times,response,color=pcolors[0])
#ax[1].plot(times,lambdas,color=pcolors[0])
#ax[0].set_xlabel("Time")
#ax[1].set_xlabel("End of Window")
#ax[0].set_ylabel(r"$x$")
#ax[1].set_ylabel(r"$|\lambda|$")
#
#for n,a in enumerate(ax.flatten()):
#    a.spines["top"].set_visible(False)
#    a.spines["right"].set_visible(False)
#    a.text(-0.1, 1.1, string.ascii_uppercase[n], transform=a.transAxes,size=12, weight='bold')
#    
#plt.tight_layout()
#plt.savefig("figure3.eps")
#plt.close()