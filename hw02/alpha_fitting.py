import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
def linear_fun(x,a,b):   #For fitting the continuum
    return a*x+b

def gaussian1(x, amp1,cen1,sigma1):  #Define gussian function
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2)))

def gaussian3(x, amp1,cen1,sigma1, amp2,cen2,sigma2, amp3,cen3,sigma3): #unit of amp: erg cm^-2 s^-1
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen1)/sigma1)**2))) + \
           amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen2)/sigma2)**2))) + \
           amp3*(1/(sigma3*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x-cen3)/sigma3)**2)))

def GaussianFitting(x, y, fr_min, fr_max, gn, peak_frac=5.):   #This function use curve_fit to fit the gussian function
    """
    Do the gaussian fitting
    ----------
    x: spectra frequency
    y: spectra flux
    fr_min: minimum frequency that you want to fitted
    fr_max: maximum frequency that you want to fitted
    gn: number of gaussian that you required
    ----------
    Returns
    - peak, center, FWHM
    """
    cri = (x >= fr_min) & (x <= fr_max)
    x, y = x[cri], y[cri]
    
    ##The initial parameters for fitting.
    if gn==1: #Hb case in the homework
        function = gaussian1
        peak1 = np.max(y)
        cent1 = fr_min + (fr_max-fr_min)/(gn*2.)
        fwhm1 = (fr_max-fr_min)/(gn*2.)
        p0=[peak1,cent1,fwhm1]
        
    if gn==3: #Ha case in this homework
        function = gaussian3
        peak1 = np.max(y)
        cent1 = fr_min + (fr_max-fr_min)/(gn*2.)
        fwhm1 = (fr_max-fr_min)/(gn*2.)
        peak2 = np.max(y)/peak_frac
        cent2 = fr_min + (fr_max-fr_min)/2.
        fwhm2 = (fr_max-fr_min)/(gn*2.)        
        peak3 = np.max(y)/peak_frac/2.
        #cent3 = fr_max - (fr_max-fr_min)/(gn*2.)
        cent3= 6545
        fwhm3 = (fr_max-fr_min)/(gn*2.)
        p0=[peak1, cent1, fwhm1, peak2, cent2, fwhm2, peak3, cent3, fwhm3]
    
    ###Fit the data using curve_fit and return the results.
    if gn==1:
        popt_gauss, pcov_gauss = curve_fit(function, x, y, p0=p0, maxfev=100000,bounds=((0,fr_min,0),(np.inf,fr_max,np.inf)))           
        perr_gauss = np.sqrt(np.diag(pcov_gauss))   

    if gn==3:
        popt_gauss, pcov_gauss = curve_fit(function, x, y, p0=p0, maxfev=100000,bounds=((0,fr_min,0,0,fr_min,0,0,fr_min,0),(np.inf,fr_max,np.inf,np.inf,fr_max,np.inf,np.inf,fr_max,np.inf)))           
        perr_gauss = np.sqrt(np.diag(pcov_gauss))   
    return popt_gauss, perr_gauss
    

###Read the spectrum data
wave,flux_den=np.genfromtxt('M82_alpha.txt',skip_header=1,usecols=(0,1),dtype=float,unpack=True)

############Ha section##############
###Set the interval for fitting. Blueside and redside are for continuum.
Ha_freq=6564
Ha_Blueside=[6483.0,6513.0]
Ha_redside=[6623.0,6653.0]
Ha_Primary=[6542.9,6595.3]  #include [NII] and Ha

###First, fit the continuum.
spec_cri_l=((Ha_Blueside[0]<wave) & (wave<Ha_Blueside[1]))
spec_cri_h=((Ha_redside[0]<wave) & (wave<Ha_redside[1]))
spec_cri=spec_cri_l+spec_cri_h
cont_popt, cont_pcov = curve_fit(linear_fun, wave[spec_cri], flux_den[spec_cri], maxfev=100000)           

###Now, fit the emission line.
flux_den_cont_sub=flux_den-linear_fun(wave,cont_popt[0],cont_popt[1])   #Subtract the continuum before fitting.
par, par_err = GaussianFitting(wave, flux_den_cont_sub, Ha_Primary[0], Ha_Primary[1], 3)

###Determine which gussian is the H alpha emission. We use the position of the gussian center to decide which one is Ha.
if abs(par[1] - Ha_freq) < abs(par[4] - Ha_freq):
    if abs(par[1] - Ha_freq) < abs(par[7] - Ha_freq):
        line_par = [abs(par[0]), par[1], abs(par[2])]
        line_par_err = [par_err[0], par_err[1], par_err[2]]
        contamination = [[abs(par[3]), par[4], abs(par[5])], [abs(par[6]), par[7], abs(par[8])]]
else:
    if abs(par[4] - Ha_freq) < abs(par[7] - Ha_freq):
        line_par = [abs(par[3]), par[4], abs(par[5])]
        line_par_err = [par_err[3], par_err[4], par_err[5]]
        contamination = [[abs(par[0]), par[1], abs(par[2])], [abs(par[6]), par[7], abs(par[8])]]
    else:
        line_par = [abs(par[6]), par[7], abs(par[8])]
        line_par_err = [par_err[6], par_err[7], par_err[8]]
        contamination = [[abs(par[0]), par[1], abs(par[2])], [abs(par[3]), par[4], abs(par[5])]]

###Plot the result
plt_cri = (wave>=Ha_Primary[0]-12) & (wave<=Ha_Primary[1]+12)
fitted_x = np.linspace(np.min(wave[plt_cri]), np.max(wave[plt_cri]), 1000)
fitted_y = gaussian3(fitted_x, *par)
fig, ax1= plt.subplots()

''' upper panel '''
ax1.plot(wave[plt_cri], flux_den[plt_cri], 'k-', drawstyle='steps-mid',lw=1)
ax1.plot(fitted_x, gaussian1(fitted_x, *line_par)+linear_fun(fitted_x,cont_popt[0],cont_popt[1]), color='r', linestyle='-', zorder=3, lw=1,label='H_alpha')
ax1.plot(fitted_x, fitted_y+linear_fun(fitted_x,cont_popt[0],cont_popt[1]), color='limegreen', linestyle='-', zorder=2, lw=1,label='fit result')
ax1.plot(fitted_x,linear_fun(fitted_x,cont_popt[0],cont_popt[1]),color='blue',label='continuum')
#ax1.plot(fitted_x, gaussian1(fitted_x, *contamination[0])+linear_fun(fitted_x,cont_popt[0],cont_popt[1]), color='gray', linestyle='--', zorder=2, dashes=(5, 1), lw=1)
#ax1.plot(fitted_x, gaussian1(fitted_x, *contamination[1])+linear_fun(fitted_x,cont_popt[0],cont_popt[1]), color='gray', linestyle='--', zorder=2, dashes=(5, 1), lw=1)
#plt.plot(wave,flux_den,marker='o')
plt.legend()
plt.title('Ha fit result',size=14)
plt.xlabel('wavelength(angstrom)')
plt.savefig('M82_new_a.png')
plt.clf()
print(line_par)

###Please note that the information of the Ha fit result is in "line_par". First number is the flux, second is the central wavelength, the third one is the dispersion.
################end of Ha section#####################


################H beta section########################
###Set the interval for fitting. Blueside and redside are for continuum.
wave,flux_den=np.genfromtxt('M82_beta.txt',skip_header=1,usecols=(0,1),dtype=float,unpack=True)
Ha_freq=4863
Ha_Blueside=[4798.9,4838.9]
Ha_redside=[4885.6,4925.6]
Ha_Primary=[4852.7,4872.7] 

###First, fit the continuum.
spec_cri_l=((Ha_Blueside[0]<wave) & (wave<Ha_Blueside[1]))
spec_cri_h=((Ha_redside[0]<wave) & (wave<Ha_redside[1]))
spec_cri=spec_cri_l+spec_cri_h
cont_popt, cont_pcov = curve_fit(linear_fun, wave[spec_cri], flux_den[spec_cri], maxfev=100000)           

###Now, fit the emission line.
flux_den_cont_sub=flux_den-linear_fun(wave,cont_popt[0],cont_popt[1])   #Subtract the continuum before fitting.
par, par_err = GaussianFitting(wave, flux_den_cont_sub, Ha_Primary[0], Ha_Primary[1], 1)

###store the fit result
line_par = [par[0], par[1], par[2]]
line_par_err = [par_err[0], par_err[1], par_err[2]]

###Plot the result
plt_cri = (wave>=Ha_Primary[0]-12) & (wave<=Ha_Primary[1]+12)
fitted_x = np.linspace(np.min(wave[plt_cri]), np.max(wave[plt_cri]), 1000)
fitted_y = gaussian1(fitted_x, *par)
fig, ax1= plt.subplots()

''' upper panel '''
ax1.plot(wave[plt_cri], flux_den[plt_cri], 'k-', drawstyle='steps-mid',lw=1)
ax1.plot(fitted_x, gaussian1(fitted_x, *line_par)+linear_fun(fitted_x,cont_popt[0],cont_popt[1]), color='r', linestyle='-', zorder=3, lw=1,label='H_alpha')
ax1.plot(fitted_x, fitted_y+linear_fun(fitted_x,cont_popt[0],cont_popt[1]), color='limegreen', linestyle='-', zorder=2, lw=1,label='fit result')
ax1.plot(fitted_x,linear_fun(fitted_x,cont_popt[0],cont_popt[1]),color='blue',label='continuum')
#ax1.plot(fitted_x, gaussian1(fitted_x, *contamination[0])+linear_fun(fitted_x,cont_popt[0],cont_popt[1]), color='gray', linestyle='--', zorder=2, dashes=(5, 1), lw=1)
#ax1.plot(fitted_x, gaussian1(fitted_x, *contamination[1])+linear_fun(fitted_x,cont_popt[0],cont_popt[1]), color='gray', linestyle='--', zorder=2, dashes=(5, 1), lw=1)
#plt.plot(wave,flux_den,marker='o')
plt.legend()
plt.title('Hb fit result',size=14)
plt.xlabel('wavelength(angstrom)')
plt.savefig('M82_new_b.png')
print(line_par)

#############################Q5 end####################################
