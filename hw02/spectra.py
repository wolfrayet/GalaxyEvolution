import numpy as np
import matplotlib.pyplot as plt

fname = 'NGC253_HI.txt'

vel, fv, poly = np.loadtxt(fname, skiprows=1, \
        usecols=(0,2,3),unpack=True)

#index = np.where(((vel>-1000) & (vel<-100))|((vel>1000) & (vel<2000)))
#continuum = np.mean(fv[index])
#sigma = np.std(fv[index])
#print('continuum:', continuum, sigma)
fv = fv - poly
index = np.where(((vel>-1000) & (vel<-200))\
        |((vel>1000) & (vel<1500)))[0]
continuum = np.std(fv[index])

index = np.where(((vel>-100) & (vel<500)) & (fv-continuum>0))[0]
print('start:',vel[index[0]])
print('end:',vel[index[-1]])
print('line width:',vel[index[-1]]-vel[index[0]])

