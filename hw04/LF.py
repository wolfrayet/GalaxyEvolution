import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
plt.rc('font', size=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

def phi_L(x,a):
    return x**a*np.exp(-x)

alpha = [-0.5, -1.0, -1.5]
x = np.logspace(-2.0,1.0,num=100)
for a in alpha:
    phi = 1./integrate.quad(phi_L, 1e-3, np.inf, args=(a),limit=100)[0]
    L_CDF = phi_L(x,a)*x*phi
    plt.plot(np.log10(x), np.log10(L_CDF), \
            label=r'$\alpha$={:.1f}'.format(a))

plt.legend()
plt.grid()
plt.xlabel('$log_{10}\,L\,/\,L^{*}$')
plt.ylabel('$log_{10}\,\phi(L) (L/L^{*})$')
plt.savefig('LF.pdf')
