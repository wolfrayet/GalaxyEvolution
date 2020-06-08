import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import astropy.constants as const
import astropy.units as u
from astropy.cosmology import LambdaCDM
from scipy.spatial import cKDTree
from scipy.stats import poisson
# plot setting
plt.rc('font', size=16)

# files
primary = 'primary'
secondary = 'secondary'
#primary = 'primary_19'
#secondary = 'secondary_19'

# parameters
H0 = 67.8 * u.km / u.s / u.Mpc
Om0 = 0.308
Ode0 = 0.692
c = const.c.to('km/s')
cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)

thres = 500 * u.km / u.s

# read file
def read_data(fname):
    data = pd.read_csv(fname)
    df = pd.DataFrame(data)
    ra = df['RA_1'][:].to_numpy()
    dec = df['DEC_1'][:].to_numpy()
    z = df['Z_1'][:].to_numpy()
    zd = df['ZDIST'][:].to_numpy()
    D4000 = df['D4000_N_SUB'][:].to_numpy()
    return ra, dec, z, zd, D4000

# count neighbor
def count_circle(xy, z, zd, kdtree, z_s, r):
    #r = r * u.Mpc
    #r_deg = r / (zd[i]*c/H0) * 180.0 / np.pi
    r_deg = r / cosmo.angular_diameter_distance(z).value * 180 / np.pi
    size = xy.shape[0]
    num = np.zeros(size, dtype=int)
    for i in range(size):
        idx = kdtree.query_ball_point(xy[i], r_deg[i])
        vel = np.abs(z_s[idx] - z[i])*c
        idx_vel = (vel <= thres)
        num[i] = np.sum(idx_vel)
    print('Finish r < {}'.format(r))
    return num-1

# count annulus
def count_annulus(num1, num2, d4000, boundary):
    idx = (num1 <= boundary[1]) & (num1 >= boundary[0])
    num = num2[idx] - num1[idx]
    print('Finish N(0-1)={}-{}, num={}'.format(boundary[0], \
            boundary[1], len(num)))
    return num, d4000[idx]

# mean and uncertainty
def mean_var(num, d4000):
    df = pd.DataFrame({'num':num, 'd4000':d4000})
    g = df.groupby('num').mean()
    num = g.index.to_numpy()
    mean = g.to_numpy().ravel()
    var = df.groupby('num').count().to_numpy().ravel()
    sigma = np.sqrt(1/var)
    return num, mean, sigma


if __name__ == '__main__':
    print(primary, secondary)
    print('vel threshold: {}'.format(thres))
    ra_p, dec_p, z_p, zd_p, d4000 = read_data(primary)
    ra_s, dec_s, z_s, _, _ = read_data(secondary)

    target = np.column_stack((ra_p*np.cos(dec_p*np.pi/180), dec_p))
    XYdata = np.column_stack((ra_s*np.cos(dec_s*np.pi/180), dec_s))
    kdtree = cKDTree(XYdata)

    num_1Mpc = count_circle(target, z_p, zd_p, kdtree, z_s, 1)
    re_num1Mpc, mean_d1Mpc, sigma_1Mpc = mean_var(num_1Mpc, d4000)
    
    num_5Mpc = count_circle(target, z_p, zd_p, kdtree, z_s, 5)
    re_num5Mpc, mean_d5Mpc, sigma_5Mpc = mean_var(num_5Mpc, d4000)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    #axs[0].scatter(num_1Mpc, d4000, alpha=0.1)
    axs[0].errorbar(re_num1Mpc, mean_d1Mpc, yerr=sigma_1Mpc, fmt='.k')
    axs[0].set_xlabel('N (0-1 Mpc)')
    axs[0].set_ylabel(r'$\rm{D_n(4000)}$')
    
    #axs[1].scatter(num_5Mpc, d4000, alpha=0.1)
    axs[1].errorbar(re_num5Mpc, mean_d5Mpc, yerr=sigma_5Mpc, fmt='.k')
    axs[1].set_xlabel('N (0-5 Mpc)')
    axs[1].set_ylabel(r'$\rm{D_n(4000)}$')
    
    plt.savefig('19thres_{}_cir.png'.format(int(thres.value)))
    plt.clf()

    num_3Mpc = count_circle(target, z_p, zd_p, kdtree, z_s, 3)
    num1_2, d1_2 = count_annulus(num_1Mpc, num_3Mpc, d4000, [1, 2])
    re1_2, meand1_2, sigma1_2 = mean_var(num1_2, d1_2)
    num8_12, d8_12 = count_annulus(num_1Mpc, num_3Mpc, d4000, [8, 12])
    re8_12, meand8_12, sigma8_12 = mean_var(num8_12, d8_12)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    #axs[0].scatter(num1_2, d1_2, alpha=0.1)
    axs[0].errorbar(re1_2, meand1_2, yerr=sigma1_2, fmt='.k')
    axs[0].set_xlabel('N (1-3 Mpc)')
    axs[0].set_ylabel(r'$\rm{D_n(4000)}$')
    axs[0].text(0.5, 0.1, 'N(0-1 Mpc)=1-2',\
            verticalalignment='center', horizontalalignment='center',\
            transform=axs[0].transAxes)

    #axs[1].scatter(num8_12, d8_12, alpha=0.1)
    axs[1].errorbar(re8_12, meand8_12, yerr=sigma8_12, fmt='.k')
    axs[1].set_xlabel('N (1-3 Mpc)')
    axs[1].set_ylabel(r'$\rm{D_n(4000)}$')
    axs[1].text(0.5, 0.1, 'N(0-1 Mpc)=8-12',\
            verticalalignment='center', horizontalalignment='center',\
            transform=axs[1].transAxes)

    plt.savefig('19thres_{}_anu.png'.format(int(thres.value)))

