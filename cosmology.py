from astropy.cosmology import LambdaCDM
import numpy as np

H0 = 67.8
Om0 = 0.308
Ode0 = 0.692

cosmos = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)

print('enter redshift:')
z = float(input())

LumDist = cosmos.luminosity_distance(z).value
AngDist = cosmos.angular_diameter_distance(z).value

print('Luminosity Distance (Mpc):',LumDist)
print('Angular Distance (Mpc):',AngDist)
