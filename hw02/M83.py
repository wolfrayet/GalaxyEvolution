import numpy as np
import matplotlib.pyplot as plt

# figure format -------------------------------------------
plt.rc('font', size=10)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)
plt.rc('figure', autolayout=True)

# constant ------------------------------------------------
h = 6.626e-34   # J * s
k = 1.380e-23   # J / K
c = 299792458   # m / s
T = 50          # K
nu = 450e9      # 450 GHz

fname='LuminosityDistance.txt'

z, LumDist, AngDist = np.loadtxt(fname,skiprows=1,usecols=(0,3,4),unpack=True)

L=(4.66)**2*6e-19
L0 = L*c/1.42e9/100 # at z=0.1
S = L0/(LumDist)**2*1e26*(1+z)/1.1



plt.plot(z,S)
plt.yscale('log')
plt.xlabel(r'$z$')
plt.ylabel(r'$S_{\nu}\;[mJy]$')
plt.grid()
plt.savefig('flux_M83.png')
plt.clf()

theta = 23/(AngDist*1e3)*180/np.pi*3600
plt.plot(z,theta)
plt.ylabel(r'$\theta [arcsec]$')
plt.xlabel(r'$z$')
plt.grid()
plt.savefig('angular_size.png')

index = np.where(z>=0.5)[0]
print('freq:',1.42/(1+z[index[-1]]))
print('z:', z[index[-1]])
print('flux:', S[index[-1]])
print('size ("):',theta[index[-1]])
