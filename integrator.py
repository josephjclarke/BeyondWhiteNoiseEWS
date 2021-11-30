#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 12:27:49 2021

@author: jc1147
"""

import numba
import numpy as np

@numba.njit
def white_noise(num, dt):
    return np.random.normal(0.0, np.sqrt(dt), num)


@numba.njit
def red_noise(num, dt, redness=0.9):
    wn = white_noise(num, dt)
    out = np.empty(num)

    out[0] = wn[0]
    for i in range(1, num):
        out[i] = redness * out[i - 1] + np.sqrt((1 - redness**2)) * wn[i]
    return out


@numba.njit
def integrate(times, x0, noise_colour, epsilon):
    num = times.size
    dt = np.diff(times)[0]

    noise = white_noise(num, dt) if noise_colour == "white" else red_noise(
        num, dt)

    #integrate xdot = x - x**3 / 3 - epsilon t + noise

    x = np.empty_like(times)
    x[0] = x0

    for i in range(1, num):
        j = i - 1
        dx = dt * (x[j] - x[j]**3 / 3 - epsilon * times[j] + noise[j])
        x[i] = x[j] + dx

    forcing = -epsilon * times + noise
    return forcing, x, noise

@numba.jit
def integrate_linear(times, x0):
    num = times.size
    dt = np.diff(times)[0]
    noise = 20.0*white_noise(num, dt)

    x = np.empty_like(times)
    x[0] = x0

    for i in range(1, num):
        j = i - 1
        dx = dt * (-np.tanh(times[j]) - x[j] -1 + noise[j])
        x[i] = x[j] + dx

    forcing = -np.tanh(times) + noise
    return forcing, x, noise



@numba.jit
def integrate_changing_variance(times,x0,epsilon,factor):
    num = times.size
    dt = np.diff(times)[0]
    noise = white_noise(num, dt) 

    #integrate xdot = x - x**3 / 3 - epsilon t + noise

    x = np.empty_like(times)
    x[0] = x0

    for i in range(1, num):
        j = i - 1
        dx = dt * (x[j] - x[j]**3 / 3 - epsilon * times[j] + noise[j]*(factor[j]))
        x[i] = x[j] + dx

    forcing = -epsilon * times + noise*(factor)
    return forcing, x

def integrate_changing_variance_no_tipping(times,x0,epsilon,factor):
    num = times.size
    dt = np.diff(times)[0]
    noise = white_noise(num, dt) 

    #integrate xdot = x - x**3 / 3 - epsilon t + noise

    x = np.empty_like(times)
    x[0] = x0

    for i in range(1, num):
        j = i - 1
        dx = dt * (x[j] - x[j]**3 / 3 + noise[j]*(factor[j]))
        x[i] = x[j] + dx

    forcing = noise*(factor)
    return forcing, x