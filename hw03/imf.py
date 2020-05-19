import numpy as np
C_uv = 5.60e-29

def salpeter_m(m):
    return m**(-1.35)

def total_mass(m):
    imf_m = salpeter_m(m)
    return np.trapz(m, imf_m)

low_mass = np.linspace(0.1,5,100)
high_mass  = np.linspace(5,100,100)

M_low_mass = total_mass(low_mass)
M_high_mass = total_mass(high_mass)

ratio = M_low_mass/M_high_mass
C_cor = ratio*C_uv

print('ratio: {:.4e}'.format(ratio))
print('C_cor: {:.4e}'.format(C_cor))
