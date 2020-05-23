import sys
import numpy as np
if __name__ == "__main__":
    # system input
    if len(sys.argv) != 2:
        print('Usage: python '+sys.argv[0]+' [fname]')
        sys.exit()
    else:
        fname = sys.argv[1]
    
    # parameters
    num = 50
    savename = fname+'_select.txt'
    header = 'ra  dec'
    
    ra, dec = np.loadtxt(fname, skiprows=1, unpack=True)
    index = np.random.randint(len(ra), size=num)
    ra_50 = ra[index]
    dec_50 = dec[index]
    
    np.savetxt(savename, np.transpose([ra_50, dec_50]), \
            fmt='%.6f', comments='', header=header)
