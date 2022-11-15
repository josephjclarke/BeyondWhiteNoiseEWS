import numpy as np
import scipy.integrate
import matplotlib.pyplot as plt
import matplotlib.colors
import tqdm
import scipy.signal
import statsmodels.tsa.stattools
import statsmodels.api as sm


pcolors = ["blue","red"]
linestyles = ["solid","dashed","dotted"]

plt.rcParams["figure.figsize"] = (9,6)
plt.rcParams["font.family"] = "serif"



def f_auto(x,t,mu):
    return x - x**3 /3.0 - mu
def f(x,t,epsilon):
    return f_auto(x,t,t)/epsilon

def find_equilibria(mus):
    forward_stable = np.zeros_like(mus)
    backward_stable = np.zeros_like(forward_stable)
    forward_unstable = np.zeros_like(forward_stable)
    for idx in tqdm.trange(mus.size):
        mu=mus[idx]
        sol = scipy.integrate.solve_ivp(lambda t,x:f_auto(x,t,mu),(0.0,100),[2.0],method="BDF")
        forward_stable[idx] = sol.y[0,-1]
        sol = scipy.integrate.solve_ivp(lambda t,x:f_auto(x,t,mu),(0.0,100),[-2.0],method="BDF")
        backward_stable[idx] = sol.y[0,-1]
        sol = scipy.integrate.solve_ivp(lambda t,x:f_auto(x,t,mu),(100.0,0),[0.0],method="BDF")
        forward_unstable[idx] = sol.y[0,-1]
    forward_unstable[np.abs(forward_unstable) > 100.0] = np.nan
    return forward_stable,backward_stable,forward_unstable

def window_var(x,length):
    var = np.full_like(x,np.nan)
    for i in tqdm.trange(var.size-length):
        var[i+length] = statsmodels.tsa.tsatools.detrend(x[i:i+length],order=2).var()
    return var

def window_ar(x,length):
    ar = np.full_like(x,np.nan)
    conf  = np.full((x.size,2),np.nan)
    for i in tqdm.trange(ar.size-length):
        ret = statsmodels.tsa.stattools.acf(statsmodels.tsa.tsatools.detrend(x[i:i+length],order=2),
                                            alpha=0.05,fft=True,nlags=5)
        ar[i+length] = ret[0][1]
        conf[i+length] = ret[1][1]
    return ar,conf

def window_spec(x,eta,length,dt,method="ratio"):
    assert x.shape == eta.shape
    assert length < x.size

    def fitfunction(f,ls):
        return (1/dt)**2/((2*np.pi*f)**2  + ls**2)


    ls = np.full_like(x,np.nan)
    ls_err = ls.copy()

    for i in tqdm.trange(ls.size-length):
        detx = statsmodels.tsa.tsatools.detrend(x[i:i+length],order=2)
        f,Sxx = scipy.signal.welch(detx,fs=1/dt)
        f,Sff = scipy.signal.welch(eta[i:i+length],fs=1/dt)#,detrend="linear")

        if method == "ratio":
            popt, pcov = scipy.optimize.curve_fit(fitfunction,
                                                  f[1:],
                                                  Sxx[1:]/Sff[1:],
                                                  p0=[1.0],
                                                  bounds=(0.0, np.inf))
            ls[i+length] = popt[0]
            ls_err[i+length] = pcov[0][0]
        else:
            def f2m(L):
                return Sff[1:] * (1/dt)**2/((2*np.pi*f[1:])**2  + L**2) - Sxx[1:]
            opt = scipy.optimize.least_squares(f2m,1.0,bounds=(0.0,np.inf))
            ls[i+length] = opt.x.item()
    return ls,ls_err

mus = np.linspace(-1,1,1000)
upper,lower,unstable = find_equilibria(mus)
upper[upper<0.0]=np.nan
lower[lower>0.0]=np.nan

jac = 1-upper**2
jac[np.isnan(upper)] = 1-lower[np.isnan(upper)]**2
true_lambda = np.abs(jac)
true_lambda[mus>2/3] = np.nan

epsilon = 0.01
Tstart = -1.0
Tend   = +1.0
npoints = 5000

dt = (Tend - Tstart)/npoints
xs = np.zeros(npoints)
ts = np.linspace(Tstart,Tend,npoints)
xs[0] = upper[0]
xs_red = xs.copy()

np.random.seed(0)

W = np.random.normal(scale=np.sqrt(dt),size=xs.size)
W_red = np.zeros_like(W)

r=0.99

for i in range(W_red.size-1):
    W_red[i+1] = r * W_red[i] + np.sqrt((1-r**2))*W[i]

W_red *= 0.5

for i in tqdm.trange(npoints-1):
    xs[i+1] = xs[i] + f(xs[i],ts[i],epsilon) * dt + W[i]
    xs_red[i+1] = xs_red[i] + f(xs_red[i],ts[i],epsilon) * dt + W_red[i]



for i in tqdm.trange(1,16):
    ys  = xs_red[::i]
    tss = ts[::i]
    tip = np.argmin(tss<2/3)
    win = np.argmin(tss<-0.8)
    win_long = np.argmin(tss<-0.6)
    ar,_ = window_ar(ys,win)
    ar_long,_ = window_ar(ys,win_long)
    tau,p = scipy.stats.kendalltau(tss[win:tip],ar[win:tip])
    tau_long,_ = scipy.stats.kendalltau(tss[win_long:tip],ar_long[win_long:tip])
    plt.scatter(i,tau,color="blue")
    plt.scatter(i,tau_long,color="red")

plt.scatter(np.nan,np.nan,color="blue",label="Normal window")
plt.scatter(np.nan,np.nan,color="red",label="Longer window")
plt.legend()
plt.axhline(0.0,color="black",linestyle="--")
plt.xlabel("Subsampling")
plt.ylabel("Kendall Tau")
plt.gca().set_xticks(np.arange(1,i+1))
plt.ylim(top=0.05)
plt.savefig("subsampling.png")
plt.show()
plt.close()

runs = 1000
tau_ar = np.zeros(runs)
tau_var = np.zeros(runs)
win = 500
for run in range(runs):
    print(run)
    W = np.random.normal(scale=np.sqrt(dt),size=xs.size)
    for i in range(W_red.size-1):
        W_red[i+1] = r * W_red[i] + np.sqrt((1-r**2))*W[i]
    W_red *= 0.5
    for i in range(npoints-1):
        xs_red[i+1] = xs_red[i] + f(xs_red[i],ts[i],epsilon) * dt + W_red[i]
    tip = np.argmin(xs_red>1.0)

    ar,_ = window_ar(xs_red,win)
    var = window_var(xs_red,win)

    tau_ar[run] = scipy.stats.kendalltau(np.arange(win,tip),ar[win:tip])[0]
    tau_var[run] = scipy.stats.kendalltau(np.arange(win,tip),var[win:tip])[0]

taus = np.linspace(0.15,0.4)
number_forwarned = np.zeros_like(taus)
number_forwarned_lams = np.zeros_like(taus)
for i,tau in enumerate(taus):
    number_forwarned[i] = np.logical_and(tau_ar > tau, tau_var > tau).sum()
plt.plot(taus,number_forwarned/runs,label="Conventional EWS")
plt.xlabel("Kendall Tau")
plt.ylabel("Probability of successful EWS")
plt.savefig("pofsuccess.png")
plt.show()


runs = 1000
tau_ar = np.zeros(runs)
tau_var = np.zeros(runs)
tau_lam = np.zeros(runs)
tip = 4166
win = 500
for run in range(runs):
    print(run)
    W = np.random.normal(scale=np.sqrt(dt),size=xs.size)
    for i in range(npoints-1):
        xs[i+1] = xs[i] + f(xs[i],ts[0],epsilon) * dt + W[i]
    ar,_ = window_ar(xs,win)
    var = window_var(xs,win)
    ls,_ = window_spec(xs,W,win,dt/epsilon)

    tau_ar[run] = scipy.stats.kendalltau(np.arange(win,tip),ar[win:tip])[0]
    tau_var[run] = scipy.stats.kendalltau(np.arange(win,tip),var[win:tip])[0]
    tau_lam[run] = scipy.stats.kendalltau(np.arange(win,tip),-ls[win:tip])[0]

taus = np.linspace(0.0,1.0)
number_forwarned = np.zeros((taus.size,3))
for i,tau in enumerate(taus):
    number_forwarned[i] = [(tau_ar>tau).sum(),(tau_var>tau).sum(),(tau_lam>tau).sum()]
plt.plot(taus,number_forwarned[:,0]/runs,label="AC")
plt.plot(taus,number_forwarned[:,1]/runs,label="Variance")
plt.plot(taus,number_forwarned[:,2]/runs,label="ROSA")
plt.xlabel("Kendall Tau")
plt.ylabel("Probability of False Positive")
plt.legend()
plt.savefig("false_positives_whitenoise.png")
plt.show()
runs = 1000
tau_ar = np.zeros(runs)
tau_var = np.zeros(runs)
tau_lam = np.zeros(runs)
tip = 4166
win = 500
for run in range(runs):
    print(run)
    W = np.random.normal(scale=np.sqrt(dt),size=xs.size)
    for i in range(W_red.size-1):
        W_red[i+1] = r * W_red[i] + np.sqrt((1-r**2))*W[i]
    W_red *= 0.5
    W=W_red
    for i in range(npoints-1):
        xs[i+1] = xs[i] + f(xs[i],ts[0],epsilon) * dt + W[i]
    ar,_ = window_ar(xs,win)
    var = window_var(xs,win)
    ls,_ = window_spec(xs,W,win,dt/epsilon)

    tau_ar[run] = scipy.stats.kendalltau(np.arange(win,tip),ar[win:tip])[0]
    tau_var[run] = scipy.stats.kendalltau(np.arange(win,tip),var[win:tip])[0]
    tau_lam[run] = scipy.stats.kendalltau(np.arange(win,tip),-ls[win:tip])[0]

taus = np.linspace(0.0,1.0)
number_forwarned = np.zeros((taus.size,3))
for i,tau in enumerate(taus):
    number_forwarned[i] = [(tau_ar>tau).sum(),(tau_var>tau).sum(),(tau_lam>tau).sum()]
plt.plot(taus,number_forwarned[:,0]/runs,label="AC")
plt.plot(taus,number_forwarned[:,1]/runs,label="Variance")
plt.plot(taus,number_forwarned[:,2]/runs,label="ROSA")
plt.xlabel("Kendall Tau")
plt.ylabel("Probability of False Positive")
plt.legend()
plt.savefig("false_positives_rednoise.png")
plt.show()
