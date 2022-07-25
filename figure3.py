#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 09:54:07 2022

@author: jc1147
"""
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
import scipy.signal
import statsmodels.api as sm
import statsmodels.tsa.stattools
import tqdm

pcolors = ["blue", "red"]
linestyles = ["solid", "dashed", "dotted"]

plt.rcParams["figure.figsize"] = (9, 6)
plt.rcParams["font.family"] = "serif"


def f_auto(x, t, mu):
    return x - x**3 / 3.0 - mu


def f(x, t, epsilon):
    return f_auto(x, t, t) / epsilon


def find_equilibria(mus):
    forward_stable = np.zeros_like(mus)
    backward_stable = np.zeros_like(forward_stable)
    forward_unstable = np.zeros_like(forward_stable)
    for idx in tqdm.trange(mus.size):
        mu = mus[idx]
        sol = scipy.integrate.solve_ivp(lambda t, x: f_auto(x, t, mu),
                                        (0.0, 100), [2.0],
                                        method="BDF")
        forward_stable[idx] = sol.y[0, -1]
        sol = scipy.integrate.solve_ivp(lambda t, x: f_auto(x, t, mu),
                                        (0.0, 100), [-2.0],
                                        method="BDF")
        backward_stable[idx] = sol.y[0, -1]
        sol = scipy.integrate.solve_ivp(lambda t, x: f_auto(x, t, mu),
                                        (100.0, 0), [0.0],
                                        method="BDF")
        forward_unstable[idx] = sol.y[0, -1]
    forward_unstable[np.abs(forward_unstable) > 100.0] = np.nan
    return forward_stable, backward_stable, forward_unstable


def window_spec(x, eta, length, dt, method="ratio"):
    assert x.shape == eta.shape
    assert length < x.size

    def fitfunction(f, ls):
        return (1 / dt)**2 / ((2 * np.pi * f)**2 + ls**2)

    ls = np.full_like(x, np.nan)
    ls_err = ls.copy()

    for i in tqdm.trange(ls.size - length):
        detx = statsmodels.tsa.tsatools.detrend(x[i:i + length], order=2)
        f, Sxx = scipy.signal.welch(detx, fs=1 / dt)
        f, Sff = scipy.signal.welch(eta[i:i + length],
                                    fs=1 / dt)  #,detrend="linear")

        if method == "ratio":
            popt, pcov = scipy.optimize.curve_fit(fitfunction,
                                                  f[1:],
                                                  Sxx[1:] / Sff[1:],
                                                  p0=[1.0],
                                                  bounds=(0.0, np.inf))
            ls[i + length] = popt[0]
            ls_err[i + length] = pcov[0][0]
        else:

            def f2m(L):
                return Sff[1:] * (1 / dt)**2 / (
                    (2 * np.pi * f[1:])**2 + L**2) - Sxx[1:]

            opt = scipy.optimize.least_squares(f2m, 1.0, bounds=(0.0, np.inf))
            ls[i + length] = opt.x.item()
    return ls, ls_err


def window_boers(x, length, dt):
    n = x.shape[0]
    xs = np.zeros_like(x)
    xs_err = np.zeros_like(x)
    for i in range(length):
        xs[i] = np.nan
    for i in range(n - length, n):
        xs[i] = np.nan
    for i in tqdm.trange(length, n - length):
        xw = x[i - length:i + length + 1]
        xw = xw - xw.mean()
        p0, p1 = np.polyfit(np.arange(xw.shape[0]), xw, 1)
        xw = xw - p0 * np.arange(xw.shape[0]) - p1
        dxw = xw[1:] - xw[:-1]

        xw = sm.add_constant(xw)
        model = sm.GLSAR(dxw, xw[:-1], rho=1)
        results = model.iterative_fit(maxiter=10)

        a = results.params[1]

        xs[i] = a
        xs_err[i] = results.bse[1]
    return xs / dt, xs_err / dt


def window_var(x, length):
    var = np.full_like(x, np.nan)
    for i in tqdm.trange(var.size - length):
        var[i + length] = statsmodels.tsa.tsatools.detrend(x[i:i + length],
                                                           order=2).var()
        #var[i+length] = scipy.signal.detrend(x[i:i+length]).var()
    return var


def window_ar(x, length):
    ar = np.full_like(x, np.nan)
    conf = np.full((x.size, 2), np.nan)
    for i in tqdm.trange(ar.size - length):
        ret = statsmodels.tsa.stattools.acf(statsmodels.tsa.tsatools.detrend(
            x[i:i + length], order=2),
                                            alpha=0.05,
                                            fft=True)
        #ret = statsmodels.tsa.stattools.acf(scipy.signal.detrend(x[i:i+length]),
        #                                    alpha=0.05,fft=True)
        ar[i + length] = ret[0][1]
        conf[i + length] = ret[1][1]
    return ar, conf


mus = np.linspace(-1, 1, 1000)
upper, lower, unstable = find_equilibria(mus)
upper[upper < 0.0] = np.nan
lower[lower > 0.0] = np.nan

jac = 1 - upper**2
jac[np.isnan(upper)] = 1 - lower[np.isnan(upper)]**2
true_lambda = np.abs(jac)
true_lambda[mus > 2 / 3] = np.nan

epsilon = 0.01
Tstart = -1.0
Tend = +1.0
npoints = 5000

dt = (Tend - Tstart) / npoints
xs = np.zeros(npoints)
ts = np.linspace(Tstart, Tend, npoints)
xs[0] = upper[0]

Dt = 0.1
delta = int(2 / Dt)
alpha = 0.75
Ts = np.zeros(delta + npoints * 2)
alphas = np.full_like(Ts, alpha)
alphas[-npoints:] = np.linspace(alpha, 0.95, npoints)
Ts[0] = np.random.normal()
np.random.seed(0)
for i in tqdm.trange(delta, Ts.size - 1):
    Ts[i + 1] = Ts[i] + Dt * (Ts[i] - Ts[i]**3 - alphas[i] * Ts[i - delta]
                              )  #+ 0.1*np.random.normal(scale=np.sqrt(Dt))
W_osc = 0.02 * Ts[-npoints:]
W_osc += np.random.normal(scale=0.005, size=npoints)
#W_osc += np.random.normal(scale=0.00005,size=npoints)
plt.plot(ts, W_osc)
plt.savefig("enso_forcing.pdf")
plt.show()
plt.plot(ts[:500], W_osc[:500])
plt.savefig("enso_forcing_zoomed.pdf")
plt.show()
for i in tqdm.trange(npoints - 1):
    xs[i + 1] = xs[i] + f(xs[i], ts[i], epsilon) * dt + W_osc[i]

period = 95
xs_des = np.fromiter((xs[i:i + period].mean()
                      for i in range(0, xs.size - period, period)), "double")
ts_des = ts[::period][:xs_des.size]
tip = 4166  #np.argmin(xs>1.0)
tip_des = np.argmin((ts_des - 2 / 3)**2)  #np.argmin(xs_des>1.0)

win = 500
win_des = win // period
ls_rosa, _ = window_spec(xs, W_osc, win, dt / epsilon, method="difference")
ls_boers, _ = window_boers(xs, win, dt / epsilon)
ls_boers_des, _ = window_boers(xs_des, win_des, period * dt / epsilon)
ls_boers = -ls_boers
ls_boers_des = -ls_boers_des

var = window_var(xs, win)
ar, _ = window_ar(xs, win)


def running_tau(t, x):
    out = np.full(x.size, np.nan)
    out[win:] = np.fromiter((scipy.stats.kendalltau(t[win:i], x[win:i])[0]
                             for i in range(win, x.size)), "double",
                            x.size - win)
    #out[:win//2] = np.nan
    return out


np.random.seed(0)
xs = np.zeros(npoints)
xs[0] = upper[0]
count = 10
taus_rosa = np.zeros((count, xs.size))
taus_bb = np.zeros_like(taus_rosa)
rs = np.linspace(0.99, 0.0, xs.size)
W_ns = xs.copy()
for c in range(count):
    print(f"On iteration {c+1} of {count}")
    W = np.random.normal(scale=np.sqrt(dt), size=xs.size)
    W_ns[0] = W[0]
    for i in range(W_ns.size - 1):
        W_ns[i + 1] = rs[i] * W_ns[i] + np.sqrt((1 - rs[i]**2)) * W[i]
    W_ns /= 2.0
    for i in range(npoints - 1):
        xs[i + 1] = xs[i] + f(xs[i], ts[i], epsilon) * dt + W_ns[i]
    ls_rosa, _ = window_spec(xs, W_ns, win, dt / epsilon, method="difference")
    ls_boers, _ = window_boers(xs, win, dt / epsilon)
    taus_rosa[c] += running_tau(ts, -ls_rosa)
    taus_bb[c] += running_tau(ts, ls_boers)

for c in range(count):
    plt.plot(ts[ts < 2 / 3], taus_rosa[c, ts < 2 / 3], color="blue", alpha=0.2)
    plt.plot(ts[ts < 2 / 3], taus_bb[c, ts < 2 / 3], color="red", alpha=0.2)

plt.plot(ts[ts < 2 / 3], taus_rosa.mean(axis=0)[ts < 2 / 3], color="blue")
plt.plot(ts[ts < 2 / 3], taus_bb.mean(axis=0)[ts < 2 / 3], color="red")
plt.xlabel(r"$t$")
plt.ylabel(r"$\tau$")
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().set_rasterized(True)
plt.savefig("figure3.eps")
plt.show()
