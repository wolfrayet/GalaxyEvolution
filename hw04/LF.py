import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc

plt.rc('font', size=16)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

alpha = [-0.5, -1.0, -1.5]

x = np.linspace(0.001,8,num=100)
for a in alpha:
    L_CDF = sc.gammainc(a+2,x)
    plt.plot(x, L_CDF, label=r'$\alpha$={:.1f}'.format(a))

plt.legend()
plt.grid()
plt.xlabel('$L / L^*$')
plt.ylabel('$L_{tot}(<L) / L_{tot}$')
plt.savefig('LF.pdf')
