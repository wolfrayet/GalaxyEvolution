import numpy as np
from astropy.modeling.blackbody import blackbody_nu
from astropy import units as u
from astropy import constants as const
import matplotlib.pyplot as plt

T = 50 * u.K
c = const.c.cgs
nu = np.logspace(6,14,1000) * u.Hz

redshift = [0.1, 0.5, 1.0, 1.5, 2.0]

for z in redshift:
    freq = nu*(1+z)
    intensity = blackbody_nu(in_x=freq, temperature=T)*(1+z)
    lam = (c/freq).to(u.micron)
    plt.plot(lam,intensity,label='{:.1f}'.format(z))


plt.legend()
plt.xlabel(r'$\lambda$ [{0:latex_inline}]'.format(lam.unit))
plt.ylabel('[{0:latex_inline}]'.format(intensity.unit))
plt.xscale('log')
plt.yscale('log')
plt.xlim(10,1e3)
plt.ylim(1e-12,1e-8)
plt.show()
