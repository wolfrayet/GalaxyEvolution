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

z, LumDist = np.loadtxt(fname,skiprows=1,usecols=(0,3),unpack=True)
plt.plot(z,LumDist)
plt.xlabel(r'$z$')
plt.ylabel('Luminosity Distance (Mpc)')
plt.savefig('LumDist.png')
plt.clf()

flux = 1e-26*(nu*(1+z))**4.5/LumDist**2
plt.plot(z,np.log10(flux),label='Rayleigh-Jeans')
plt.xlabel(r'$z$')
plt.ylabel(r'$log_{10}\,S_{\nu}$')
plt.savefig('flux_z_RJ.png')
plt.clf()

flux_BB = 1e-26*(nu*(1+z))**5.5/(np.exp(h*nu*(1+z)/T/k)-1)/LumDist**2
plt.plot(z,np.log10(flux_BB),label='Planck')
plt.xlabel(r'$z$')
plt.ylabel(r'$log_{10}\,S_{\nu}$')
plt.savefig('flux_z_BB.png')
plt.clf()

N = flux[-1]/flux_BB[-1]

plt.plot(z,np.log10(flux),label='Rayleigh-Jeans')
plt.plot(z,np.log10(flux_BB*N),label='Planck')
plt.legend()
plt.xlabel(r'$z$')
plt.ylabel(r'$log_{10}\,S_{\nu}$')
plt.savefig('flux_z.png')



