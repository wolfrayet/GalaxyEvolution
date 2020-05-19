import numpy as np
from astropy.modeling.blackbody import blackbody_nu
from astropy import units as u
from astropy import constants as const

# parameters
Teff = 43650 * u.K
L_O5 = 2.4e5 * const.L_sun.cgs
M = 40 * u.Msun
tau = 4.3 * 1e6 *u.yr


# constants
c = const.c.cgs
sigma = const.sigma_sb.cgs # Stephan-Boltzmann

# UV range (freq)
uv_low = (c.to(u.AA/u.s)/(2800 * u.AA)).to(u.Hz)
uv_up = (c.to(u.AA/u.s)/(1500 * u.AA)).to(u.Hz)
BW = uv_up - uv_low

# BB evaluation
L_all = (sigma*Teff**4).to(u.erg/u.cm**2/u.s)
freq_uv = np.linspace(uv_low, uv_up, 200).to(u.Hz)
I_uv = blackbody_nu(in_x=freq_uv, temperature=Teff)
L_uv = np.trapz(x=freq_uv, y=I_uv)*np.pi*u.sr
frac = L_uv/L_all

print('bolometric: {:.4e}'.format(L_all))
print('UV(1500-2800A): {:.4e}'.format(L_uv))
print(L_uv/BW)
print('ratio(UV to bol): {:.4f}'.format(frac))

freq_all = np.linspace(1e-6, 1.5e4, 1000)*1e12*u.Hz
I_all = blackbody_nu(in_x=freq_all, temperature=Teff)
L_all = np.trapz(x=freq_all, y=I_all)*np.pi*u.sr
frac = L_uv/L_all
print(frac)
# conversion
L_star = L_O5 * frac / BW
SFR = M/tau
C_star = SFR/L_star

print('UV bandwidth: {:.4e}'.format(BW))
print('UV luminosity: {:.4e}'.format(L_star))
print('SFR: {:.4e}'.format(SFR))
print('C: {:.4e}'.format(C_star))
