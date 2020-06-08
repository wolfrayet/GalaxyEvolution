import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.cosmology import LambdaCDM
# input
fname = 'clustercoma'

# parameters
K_r = -0.19     # K correction red galaxies
K_b = -0.16     # K correction blue galaxies
H0 = 67.8
Om0 = 0.308
Ode0 = 0.692

# absolute magnitude
def abs_mag(m, dL, K):
    return m - 5*np.log10(dL) - 25 + K

def div_james(x):
    return -0.018*(x+21)**2 - 0.137*(x+21)+2.2

def div_baldry(x):
    return 2.06 - 0.244*np.tanh((x+20.07)/1.09)

def color(index, u_r, r, dL, K):
    r_new = r[index]
    dl = dL[index]
    Mr = abs_mag(r_new, dl, K)
    return u_r[index], Mr

def LF(M, phi, a, M_star):
    temp = 10**(0.4*(M_star-M))
    return phi*0.4*np.log(10)*temp**(a+1)*np.exp(-temp)

# read data
data = pd.read_csv(fname)
df = pd.DataFrame(data)
u = df['dered_u'][:].to_numpy()
r = df['dered_r'][:].to_numpy()
z = df['z'][:].to_numpy()

cosmos = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
dL = cosmos.luminosity_distance(z).value        # Mpc

u_r = u - r
Mr = abs_mag(r, dL, K_r)
x = np.linspace(np.min(Mr), np.max(Mr))
divide_baldry = div_baldry(x)
divide_james = div_james(x)
'''
# CMD 
plt.scatter(Mr, u-r)
plt.plot(x, divide_baldry, 'k-', label='Baldry+2004')
plt.plot(x, divide_james, 'k--', label='James+2017')
plt.xlim(-24, -15)
plt.ylim(0.3, 3.3)
plt.gca().invert_xaxis()
plt.xlabel(r'$\mathrm{M_r}$')
plt.ylabel(r'$u-r$')
plt.legend(loc=4)
plt.savefig('CMD_coma.png')
plt.clf()
'''
# divide james
index_b = np.where(u_r <= div_james(Mr))
index_r = np.where(u_r > div_james(Mr))
blue_color, blue_Mr = color(index_b, u_r, r, dL, K_b)
red_color, red_Mr = color(index_r, u_r, r, dL, K_r)
plt.scatter(blue_Mr, blue_color, label='b')
plt.scatter(red_Mr, red_color, label='r')
plt.plot(x, divide_james, 'k--', label='James+2017')
plt.xlim(-24, -15)
plt.ylim(0.3, 3.3)
plt.gca().invert_xaxis()
plt.legend(loc=4)
plt.xlabel(r'$\mathrm{M_r}$')
plt.ylabel(r'$u-r$')
plt.savefig('CMD_coma_corr.png')
plt.clf()

## LF
lf = LF(x, 60, -1.05, -21.28)
plt.plot(x, lf, 'k-' ,label='Blanton+ 2003')
lf = LF(x, 40, -1.12, -22)
plt.plot(x, lf, 'k-.' ,label='fit by eye')
tot_Mr = np.concatenate((blue_Mr,red_Mr))
plt.hist(tot_Mr, bins=20, color='#bfbfbf')
plt.xlabel(r'$\mathrm{M_r}$')
plt.yscale('log')
plt.legend(loc=2)
plt.savefig('lf_blanton.png')
plt.clf()

'''
lf = LF(x, 30, -1.18, -22.63)
plt.plot(x, lf, 'k-' ,label='Mobasher+ 2003')
lf = LF(x, 40, -1.12, -22)
plt.plot(x, lf, 'k-.' ,label='fit by eye')
tot_Mr = np.concatenate((blue_Mr,red_Mr))
plt.hist(tot_Mr, bins=20, color='#bfbfbf')
plt.hist(blue_Mr, bins=20, color='#2291e0', alpha=0.3, label='b')
plt.hist(red_Mr, bins=20, color='#f55f80', alpha=0.3, label='r')
plt.xlabel(r'$\mathrm{M_r}$')
plt.yscale('log')
plt.legend(loc=2)
plt.savefig('lf_james.png')
plt.clf()
lf = LF(x, 30, -1.18, -22.63)
plt.plot(x, lf, 'k-' ,label='Mobasher+ 2003')
lf = LF(x, 40, -1.12, -22)
plt.plot(x, lf, 'k-.' ,label='fit by eye')
tot_Mr = np.concatenate((blue_Mr,red_Mr))
plt.hist(tot_Mr, bins=20, color='#bfbfbf')
plt.hist(blue_Mr, bins=20, color='#2291e0', alpha=0.3, label='b')
plt.hist(red_Mr, bins=20, color='#f55f80', alpha=0.3, label='r')
plt.xlabel(r'$\mathrm{M_r}$')
plt.yscale('log')
plt.legend(loc=2)
plt.savefig('lf_james.png')
plt.clf()
'''
'''
plt.hist(blue_Mr, bins=10, color='#2291e0')
plt.xlabel(r'$\mathrm{M_r}$')
plt.savefig('lf_blue_james.png')
plt.clf()
plt.hist(red_Mr, bins=10, color='#f55f80')
plt.xlabel(r'$\mathrm{M_r}$')
plt.savefig('lf_red_james.png')
plt.clf()

# divide baldry
index_b = np.where(u_r <= div_baldry(Mr))
index_r = np.where(u_r > div_baldry(Mr))
blue_color, blue_Mr = color(index_b, u_r, r, dL, K_b)
red_color, red_Mr = color(index_r, u_r, r, dL, K_r)
plt.scatter(blue_Mr, blue_color, label='b')
plt.scatter(red_Mr, red_color, label='r')
plt.plot(x, divide_baldry, 'k-', label='Baldry+2004')
plt.xlim(-24, -15)
plt.ylim(0.3, 3.3)
plt.gca().invert_xaxis()
plt.legend(loc=4)
plt.xlabel(r'$\mathrm{M_r}$')
plt.ylabel(r'$u-r$')
plt.savefig('CMD_coma_bal.png')
plt.clf()
## LF
tot_Mr = np.concatenate((blue_Mr,red_Mr))
plt.hist(tot_Mr, bins=20, color='#a2a6a8')
plt.xlabel(r'$\mathrm{M_r}$')
plt.savefig('lf_bal.png')
plt.clf()
plt.hist(blue_Mr, bins=10, color='#2291e0')
plt.xlabel(r'$\mathrm{M_r}$')
plt.savefig('lf_blue_bal.png')
plt.clf()
plt.hist(red_Mr, bins=10, color='#f55f80')
plt.xlabel(r'$\mathrm{M_r}$')
plt.savefig('lf_red_bal.png')
plt.clf()
'''
