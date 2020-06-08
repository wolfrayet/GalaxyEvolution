import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import LambdaCDM
from scipy.integrate import simps
plt.rc('font', size=14)
# parameters
H0 = 67.8
Om0 = 0.398
Ode0 = 0.692
c = const.c
G = const.G
Msun = const.M_sun
K_r = -0.19     # K correction red galaxies
K_b = -0.16     # K correction blue galaxies
r_sun = 4.76    # solar magnitude at r-band at z=0.1

zc = 0.0231                 # center of coma
z_cmv = 0.0233              # comoving
R = 1 * const.pc * 1e6

# read data
def read_data(fname):
    data = pd.read_csv(fname)
    df = pd.DataFrame(data)
    z = df['z'][:].to_numpy()
    r = df['dered_r'][:].to_numpy()
    u = df['dered_u'][:].to_numpy()
    u_r = u - r
    return z, r, u_r

# redshift correct
def z_corr(z, zc):
    return (1+z) / (1+zc) - 1

# select
def select_coma(z_pec, z_std):
    return (z_pec <= 2*z_std) & (z_pec >= -2*z_std)

def div_baldry(x):
    return 2.06 - 0.244*np.tanh((x+20.07)/1.09)

def div_yu(x):
    return -0.018*(x+21)**2 - 0.137*(x+21)+2.2

# absolute magnitude
def abs_mag(m, dL, K):
    return m - 5*np.log10(dL) - 25 - K

# luminosity in solar luminosity
def luminosity(Mr):
    return 10**(-(Mr - r_sun)/2.5)

def MF(M, phi, a, M_star):
    temp = 10**(0.4*(M_star-M))
    return phi*0.4*np.log(10)*temp**(a+1)*np.exp(-temp)

def Lphi_faint(x, a, x_1, phi_1):
    return x*phi_1*(x/x_1)**a

def extra(a, L_1, phi_1, L_str, end_frac):
    x_2 = end_frac
    x_1 = L_1/L_str
    #x_1 = 100
    x = np.linspace(x_2, x_1, num=1000)
    L = Lphi_faint(x, a, 1., phi_1)*L_str
    return np.trapz(L, x)

def Lphi(x, a, amp):
    return amp*x**(a+1)*np.exp(-x)

def extra_all(a, amp, L_str, x1, x2):
    x = np.linspace(x1, x2, num=1000)
    L = L_str*Lphi(x, a, amp)
    return np.trapz(L, x)

# main
if __name__ == "__main__":
    # read data
    z, r, u_r = read_data('coma_yuhsiuhuang.csv')
    #z, r, u_r = read_data('clustercoma')
    
    # redshift and velocity dispersion
    z_pec = z_corr(z, zc)
    z_std = np.std(z_pec)
    vel_std = z_std * c
    print('z_std: {:.4f}    vel_std: {:.4e}'.format(z_std, \
            vel_std.to(u.km/u.s)))

    # mass
    M = 5 * vel_std**2 * R / G / Msun
    print('mass: {:.4e} Msun'.format(M))

    # select Coma cluster member
    idx = select_coma(z_pec, z_std)
    z_select = z_corr(z[idx], z_pec[idx])
    r_select = r[idx]
    u_r_select = u_r[idx]

    # correct to CMV
    zc_delta = z_corr(zc, z_cmv)
    z_select = z_corr(z_select, zc_delta)
    
    # dL and Mr
    cosmos = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
    dL = cosmos.luminosity_distance(z_select).value
    Mr = abs_mag(r_select, dL, -0.175)

    # distinguish blue and red galaxies and calculate Lr
    #idx_blue = u_r_select <= div_yu(Mr)
    idx_blue = u_r_select <= div_baldry(Mr)
    Mr_blue = abs_mag(r_select[idx_blue], dL[idx_blue], K_b)
    Mr_red = abs_mag(r_select[~idx_blue], dL[~idx_blue], K_r)
    Lr_blue = luminosity(Mr_blue)
    Lr_red = luminosity(Mr_red)
    print('BG/all:  {:3d}/{}'.format(len(Lr_blue), len(r_select)))
    print('RG/all:  {:3d}/{}'.format(len(Lr_red), len(r_select)))

    # total luminosity
    Lr_all = np.concatenate((Lr_blue, Lr_red))
    Ltot = np.sum(Lr_all)
    ratio = M/Ltot
    print('Ltot: {:.4e} Lsun, ratio: {:.4e} Msun/Lsun'.format(Ltot,ratio))

    amp = 80
    M_str = -21
    a = -1.1
    b = -17
    bins = 20
    L_str = luminosity(M_str)
    print('a={:.1f}  phi*={}  Mstr={:.1f}  Lstr: {:.4e} Lsun'
            .format(a,amp,M_str,L_str))

    Mr_all = np.concatenate((Mr_blue, Mr_red))
    #M_hist, bin_edges = np.histogram(Mr_all, bins=bins)

    x = np.linspace(np.min(Mr_all), np.max(Mr_all))
    mf = MF(x, amp, a, M_str)
    
    plt.hist(Mr_all, bins=bins, color='#bfbfbf')
    plt.plot(x, mf, 'k-.')
    #plt.scatter(bin_edges[b], M_hist[b])
    plt.xlabel(r'$\mathrm{M_r}$')
    plt.yscale('log')
    plt.savefig('MF.png')

    
    M_1 = -20   # start of extrapolation
    L_1 = luminosity(M_1)/L_str
    print('start mag/L: {:.4f}/{:.4e} L*'.format(M_1, L_1))
    
    idx = Mr_all <= M_1
    L_bright = np.sum(Lr_all[idx])
    L_fit_bright = extra_all(a, amp, L_str, L_1, 100)

    norm = L_bright/L_fit_bright
    print('bright: {:.4e} Lsun'.format(L_bright))
    print('bright fit: {:.4e} Lsun'.format(L_fit_bright))
    print('real/fit: {:.4f}'.format(norm))

    #L_01 = extra(a, L_1, M_hist[b], L_str, 0.1)*norm
    L_01 = extra_all(a, amp, L_str, 0.1, L_1)*norm
    L_01 += L_bright
    ratio = M/L_01
    print('L01_tot: {:.4e} Lsun, ratio: {:.4e} Msun/Lsun'.format(L_01,ratio))
    
    #L_001 = extra(a, L_1, M_hist[b], L_str, 0.01)*norm
    L_001 = extra_all(a, amp, L_str, 0.01, L_1)*norm
    L_001 += L_bright
    ratio = M/L_001
    print('L001_tot: {:.4e} Lsun, ratio: {:.4e} Msun/Lsun'.format(L_001,ratio))

    

