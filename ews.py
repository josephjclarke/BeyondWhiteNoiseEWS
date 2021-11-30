#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:28:15 2021

@author: jc1147
"""

import scipy.signal
import numpy as np
import pandas as pd

def equilibrium(ts, epsilon):
    def eq_helper(t):
        x0 = np.sqrt(3) if epsilon * t < 2 / 3.0 else -np.sqrt(3)
        return scipy.optimize.root(lambda x: x - x**3 / 3 - epsilon * t,x0).x.item()
    return np.fromiter((eq_helper(t) for t in ts),dtype="double",count=ts.size)

def traditional_ews(sig,win):
    s = pd.Series(sig)
    v = s.rolling(win).apply(lambda x: scipy.signal.detrend(x).var())
    a = s.rolling(win).apply(lambda x: pd.Series(scipy.signal.detrend(x)).autocorr())
    return v,a

def fitfunction(frequencies, ls):
    return 1 / (frequencies**2 + ls**2)


def lambda_from_spectrum(forcing, response, window, fs):
    lambdas = np.full_like(forcing, np.nan)

    for i in range(lambdas.size - window):
        dr = response[i:i + window]
        df = forcing[i:i + window]

        f, Pxx_f = scipy.signal.periodogram(df, fs=fs, detrend="linear")
        f, Pxx_r = scipy.signal.periodogram(dr, fs=fs, detrend="linear")
        popt, pcov = scipy.optimize.curve_fit(fitfunction,
                                              f[1:],
                                              Pxx_r[1:] / Pxx_f[1:],
                                              p0=[1.0],
                                              bounds=(0.0, np.inf))

        lambdas[i + window] = popt[0]

    return lambdas
