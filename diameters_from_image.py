#!/usr/bin/python

"""
A small analysis program for images taken with the LASER spot and a reference grid. 
The image needs to have been transformed to FITS format (preferably the green color part)

The program read the fits-file, and fits an elliptic Gaussian to the LASER spot
It then subtracts the background (calculated from beyond an ellipse of "securityfac" times the 
fitted half lengths and widths and normalizes the content. 

The background-subtracted image is then integrated in concentric ellipses around the fitted center, 
and the image integral contained outside the ellipses calculated. 

The result is plot of half width/length vs. image integral, from which one can read off 
the beam diameters at the reference value of 1/e^2 (indicated by a horizontal line)

CONFIGURATION:
1) Open the image first with Gimp and select a suitable cutout (containing the full Laser image and some constant background)
2) Write the full image name in the parameter "file"
3) Use the cursor to pick the edge points of the cutout
4) Write these edge coordinates into the variables "rangex1, rangex2, rangey1 and rangey2"
5) Use the reference grid to calculate the conversion from points to mm (always using Gimp)
6) Write the conversion into the parameter "conv"
7) Run the program
"""
from scipy.optimize import leastsq
from scipy import interpolate, arange, zeros
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib import cm
import math
from pylab import figure, axes, pie, title, show
import sys
#
# user configuration: 
#
# image file in FITS format (green):
#file='./_DSC0530-G.fits'
#file='/Users/markusgaug/CTA/LIDAR/Measurements/20170410_Laser/IMG_0767-G.fits'
# image cutout (from gimp) coordinates
#rangex1=2000
#rangex2=2300
#rangey1=760
#rangey2=1100
# conversion from points to mm (please use GIMP and the reference grid on the picture!
#conv   = 0.844921    # 1./8.31    # conversion from points to mm
conv = 0.0353 # from millimetric paper

OUT_OF_RANGE_VALUE = 99999999


def roundto(x,sigfigs=2) -> str:
    '''
    Suppose we want to show 2 significant figures. Implicitly we want to show 3 bits of information:
    - The order of magnitude
    - Significant digit #1
    - Significant digit #2
    '''
    order_of_magnitude = np.floor(np.log10(np.abs(x)))

    # Because we get one sigfig for free, to the left of the decimal
    decimals = (sigfigs - 1)

    x /= np.power(10, order_of_magnitude)
    x  = np.round(x, decimals)
    x *= np.power(10., order_of_magnitude)

    return x


def gaussian(height, center_x, center_y, width_x, width_y, theta, offset):
    """Returns a gaussian function with the given parameters"""
    width_x = float(width_x)
    width_y = float(width_y)
    a = (np.cos(theta) ** 2) / (    width_x ** 2) + (np.sin(theta) ** 2) / (    width_y ** 2)  
    b = -(np.sin(2 * theta)) / (2 * width_x ** 2) + (np.sin(2 * theta))  / (2 * width_y ** 2)   
    c = (np.sin(theta) ** 2) / (    width_x ** 2) + (np.cos(theta) ** 2) / (    width_y ** 2)  

    OUT_OF_RANGE = lambda x, y: OUT_OF_RANGE_VALUE
    #    print (height, center_x, center_y, width_x, width_y, theta, offset)
    if (height<0):
        return OUT_OF_RANGE
    if (center_x < 0.):
        return OUT_OF_RANGE 
    if (center_y < 0.):
        return OUT_OF_RANGE 
    if (width_x < 0.):
        return OUT_OF_RANGE 
    if (width_y < 0.):
        return OUT_OF_RANGE 
    if (offset < 0.):
        return OUT_OF_RANGE 
    if (theta > 3.*np.pi):
        return OUT_OF_RANGE 
    return lambda x, y: offset + height * np.exp( - (a* (center_x - x) ** 2 + c * (center_y - y) ** 2 + 2*b * (center_x - x)*(center_y-y)) / 2.)


def moments(data):
    """Returns (height, x, y, width_x, width_y, theta, offset)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = data.sum()
    print ("total:", total)
    X, Y = np.indices(data.shape)
    print ("Shape:", data.shape)
    print ("X=", X)
    print ("Y=", Y)
    x = (X * data).sum() / total
    y = (Y * data).sum() / total
    print ("x=", x)
    print ("y=", y)
    
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size) - y) ** 2 * col).sum() / col.sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size) - x) ** 2 * row).sum() / row.sum())
    height = data.max()
    offset = total / (col.size*row.size)

    print('from moments: height=', height, 'x=', x, 'y=', y, 'width_x=', width_x, 'width_y=', width_y, 'theta=', -45*np.pi/180, 'offset=', offset)
    y = 910.
    x = 475.
    width_y = 35./2
    width_x = 37./2
    return height, x, y, 2.*width_x, width_y, -90*np.pi/180, offset


def fitgaussian(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    params = moments(data)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(data.shape)) -
                                       data)
    p, success = leastsq(errorfunction, params)
    return p, errorfunction


def run_script(file,rangey1,rangey2,rangex1,rangex2):

    print('ranges: ',rangey1,rangey2,rangex1,rangex2)

    outfile = str(file).replace(".fits","")
    outfile1 = outfile+"_1.pdf"
    print ("outfile=",outfile1)

    beamdiameterdef = 1 #/math.e**2 # definition of beam diameter
    securityfac_bg  = 9              # times the fitted half widths beyond which pixels are counted as background
    securityfac_sig = 2.35           # times the fitted half widths beyond which pixels are counted as signal    
    fac             = 0.025          # required precision in units ellipse half width and length
    upx             = 142*conv       # upper range of plot (in mm)
    lox             = -0*conv       # lower range of plot (in mm)

    idx    = rangex2-rangex1
    idy    = rangey2-rangey1

    dim = int(0.5*(idx+idy)*4)

    hdulist = fits.open(file)
    print (hdulist.info)
    hdu = hdulist[1]
    print (hdu)
    
    print (hdu.shape)
    
    colors = ('red', 'red', 'red', 'black')
    titles = ('linear, fitted', 'log-scale', 'background-subtracted', 'light spot and off-region')
    f, axarr = plt.subplots(2, 2)
    
    x = 0.
    y = 0.
    width_x = 0.
    width_y = 0.
    
    scidata = hdu.data
    scidata_sl = scidata[rangey1:rangey2,rangex1:rangex2]
    print(scidata_sl)

    # replace the negative values by means of surroundings
    mark_x = np.where( scidata_sl<0 )[0]
    mark_y = np.where( scidata_sl<0 )[1]    
    for x, y in zip(mark_x,mark_y):
        slice = scidata_sl[max(0, x-2):min(scidata_sl.shape[0]-1,x+2), max(0,y-2):min(scidata_sl.shape[1]-1,y+2)] # assuming you want 5x5 square
        scidata_sl[x,y] = np.mean([i for i in slice.flatten() if i > 0])  # threshold is 0

    #scidata_sl[scidata_sl < 0] = 256-118.769
    
    axarr[0, 0].imshow(scidata_sl, origin='lower', cmap='gnuplot')
    axarr[0, 0].set_title(titles[0], color=colors[0], fontsize=10)
    
    params, errf = fitgaussian(scidata_sl)
    fit = gaussian(*params)
    
    arr = fit(*np.indices(scidata_sl.shape))
    print(arr.shape, scidata_sl.shape)
    chi2 = np.sum(np.power(arr-scidata_sl,2))/(scidata_sl.size-params.size)

    print('                      height,          x,              y,        width_x,         width_y,        theta,     offset,       chi2')
    print('Fitted parameters: ', [ "{:f} ".format(x) for x in params], chi2 )

    axarr[0, 0].contour(fit(*np.indices(scidata_sl.shape))) #, cmap=plt.cm.copper)
    (height, x, y, width_x, width_y, theta, offset) = params

    axarr[0, 0].text(0.,0.9, r'$\sigma_x$={:3.1f}, $\sigma_y$={:3.1f}, $\alpha$={:.0f}$^\circ$'.format(width_y, width_x, theta*57.3,prec='.1'),
                     verticalalignment='bottom', horizontalalignment='left',
                     transform=axarr[0, 0].transAxes,
                     color='white', fontsize=8)       
    num = 0
    offset = 0.
        
    #scidata = hdu.data
    #scidata_sl = scidata[rangey1:rangey2,rangex1:rangex2]
    scidata_sl2 = np.array(scidata_sl) + 0.

    summ = 0.
    numsum = 0

    # for rotated ellipse equations, see e.g. https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate
    a_bg = (np.cos(theta) ** 2) / (    (securityfac_bg* width_x) ** 2) + (np.sin(theta) ** 2) / (    (securityfac_bg* width_y) ** 2)
    b_bg = -(np.sin(2 * theta)) / (2 * (securityfac_bg* width_x) ** 2) + (np.sin(2 * theta))  / (2 * (securityfac_bg* width_y) ** 2)
    c_bg = (np.sin(theta) ** 2) / (    (securityfac_bg* width_x) ** 2) + (np.cos(theta) ** 2) / (    (securityfac_bg* width_y) ** 2)

    for ix in range(0, idy):
        for iy in range(0, idx):
            if (np.abs(iy-int(y))<2 and np.abs(ix-int(x))<2):
                print ('HERERERE',ix, x, width_x, iy, y, width_y, a_bg* (ix - x) ** 2 +  c_bg * (iy - y) ** 2 + 2*b_bg* (ix - x)*(iy - y))
            #        if ((((ix - x) / width_x * 2) ** 2 + ((iy - y) / width_y * 2) ** 2) > 1):
            if ((  a_bg* (ix - x) ** 2 +  c_bg * (iy - y) ** 2 + 2*b_bg* (ix - x)*(iy - y)) > 1) and (scidata_sl2[ix,iy]<100):
                offset += scidata_sl2[ix, iy]
                if (scidata_sl2[ix,iy] < 0):
                    print ('ix=',ix,'iy=',iy,'offset=',offset,'scidata=',scidata_sl2[ix,iy])
                num += 1

    print ('sum=', offset, ' num=', num)
    offset /= num
    print ('offset=', offset)

    a = (np.cos(theta) ** 2) / (    (securityfac_sig* width_x) ** 2) + (np.sin(theta) ** 2) / (    (securityfac_sig* width_y) ** 2)
    b = -(np.sin(2 * theta)) / (2 * (securityfac_sig* width_x) ** 2) + (np.sin(2 * theta))  / (2 * (securityfac_sig* width_y) ** 2)
    c = (np.sin(theta) ** 2) / (    (securityfac_sig* width_x) ** 2) + (np.cos(theta) ** 2) / (    (securityfac_sig* width_y) ** 2)

    for ix in range(0, idy):
        for iy in range(0, idx):
            if ((  a* (ix - x) ** 2 +  c * (iy - y) ** 2 + 2*b* (ix - x)*(iy - y)) < 1):
                summ += scidata_sl2[ix, iy]
                numsum += 1

    print ('total signal=', summ, ' sigcounts=', numsum)    
    summ -= numsum * offset
    print ('total signal bg subtracted=', summ)    

    integrals  = np.zeros(dim, dtype=float)
    #integrals  = np.zeros((dim, 1), dtype=float)
    irradiance = np.zeros(dim, dtype=float)
#    w_x = np.zeros((dim, 1), dtype=float)
#    w_y = np.zeros((dim, 1), dtype=float)
#    wxx = np.zeros((dim, 1), dtype=float)
#    wyy = np.zeros((dim, 1), dtype=float)
#    aa  = np.zeros((dim, 1), dtype=float)
#    bb  = np.zeros((dim, 1), dtype=float)
#    cc  = np.zeros((dim, 1), dtype=float)
#    sfac = int(securityfac / fac) - 1

    ii = np.arange(dim)
    w_x = (ii + 1) * width_x * fac    
    w_y = (ii + 1) * width_y * fac
    #wxx = 2. * w_x * conv       # convert half-axes to full axes
    #wyy = 2. * w_y * conv       # convert half-axes to full axes
    wxx = w_x * conv       # want half-axes here
    wyy = w_y * conv       # want half-axes here
    #    print(list(zip(ii, w_x, w_y, wxx, wyy)))
    # for rotated ellipsee equations, see e.g. 
    # https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate
    aa = (np.cos(theta) ** 2) / (    w_x ** 2) + (np.sin(theta) ** 2) / (    w_y ** 2)
    bb = -(np.sin(2 * theta)) / (2 * w_x ** 2) + (np.sin(2 * theta))  / (2 * w_y ** 2)
    cc = (np.sin(theta) ** 2) / (    w_x ** 2) + (np.cos(theta) ** 2) / (    w_y ** 2)
    
    scidata_test  = np.zeros(scidata_sl2.shape , dtype=float)
    scidata_test2 = np.zeros(scidata_sl2.shape , dtype=float)

    print ("idx:",idx, "idy:",idy,"dim:",dim)

    scidata_log = np.log(scidata_sl2)
    scidata_sl2 -= offset
    scidata_sl2 /= summ
    scidata_test += 2.
    #scidata_test[scidata_sl>137] += 1.
    
    for ix in range(idy):
        print ("ix:", ix, " out of:",idy, end="\r", flush=True)
        for iy in range(idx):
            #scidata_log[ix, iy]  = np.log(scidata_sl2[ix,iy])
            #scidata_sl2[ix, iy] -= offset
            #scidata_sl2[ix, iy] /= summ

            # replaces the slow loop below
            ellipse_cond_add = np.where( (aa* (ix - x) ** 2 +  cc * (iy - y) ** 2 + 2*bb* (ix - x)*(iy - y)) < 1, scidata_sl2[ix,iy],0.)
            integrals += ellipse_cond_add
            
            #for ii in range(dim):
                #            if ((((ix - x) / w_x[ii] * 2) ** 2 + ((iy - y) / w_y[ii] * 2) ** 2) < 1):
                #if ((  aa[ii]* (ix - x) ** 2 +  cc[ii] * (iy - y) ** 2 + 2*bb[ii]* (ix - x)*(iy - y)) > 1):
                #    integrals[ii] += scidata_sl2[ix, iy]
                #if (ii==sfac):
                #    scidata_test[ix,iy] += 2.

    print ("First loop done...")
                    
    for ii in range(dim-1,0,-1):
        #print ("ii: ",ii, " integrals: ",integrals[ii])
        if (integrals[ii] < beamdiameterdef):
            for ix in range(idy):
                for iy in range(idx):
                    if ((  aa[ii]* (ix - x) ** 2 +  cc[ii] * (iy - y) ** 2 + 2*bb[ii]* (ix - x)*(iy - y)) > 1):
                        scidata_test[ix,iy] += 1.
            break

    for ix in range(idy):
        for iy in range(idx):
            if ((  a_bg * (ix - x) ** 2 +  c_bg * (iy - y) ** 2 + 2*b_bg* (ix - x)*(iy - y)) > 1):
                scidata_test[ix,iy] += 1.

    print ("Second loop done...")
        
    irradiance[0] = (1. - integrals[0]) / (wxx[0]*wyy[0])
    for ii in range(1, dim-1):
        irradiance[ii] = (integrals[ii-1] - integrals[ii]) / (wxx[ii]*wyy[ii]-wxx[ii-1]*wyy[ii-1])

    for ii in range(0, dim):
        irradiance[dim-ii-1] /= irradiance[0]

    irradiance[dim-1] = 0;

    #for ii in range(0, dim):
    #    print ("wx=%04.3f" % w_x[ii], "wy=%04.3f" % w_y[ii], " int=%.04f" % integrals[ii], " irr=%.04f" % irradiance[ii])


    axarr[0, 1].imshow(scidata_log, origin='lower', cmap='gnuplot')
    axarr[0, 1].set_title(titles[1], color=colors[1], fontsize=10)

    axarr[1, 0].imshow(scidata_sl2, origin='lower', cmap='gnuplot')
    axarr[1, 0].set_title(titles[2], color=colors[2], fontsize=10)
    
    axarr[1, 1].imshow(scidata_test, origin='lower',cmap='gnuplot')
    axarr[1, 1].set_title(titles[3], color=colors[3], fontsize=10)

    f.tight_layout()
    f.savefig(outfile+"_1.pdf")

    # Interpolation object
    fiy = interpolate.interp1d(integrals, wyy, assume_sorted = False)
    fix = interpolate.interp1d(integrals, wxx, assume_sorted = False)
    fry = interpolate.interp1d(irradiance,wyy, assume_sorted = False)
    frx = interpolate.interp1d(irradiance,wxx, assume_sorted = False)
    #print ("ee=", beamdiameterdef, " interp=", fiy(beamdiameterdef))

    diamy = float(fiy(beamdiameterdef))
    diamx = float(fix(beamdiameterdef))
    diary = float(fry(beamdiameterdef))
    diarx = float(frx(beamdiameterdef))
    
    print (diamx, diamy)

    g,f2 = plt.subplots(1,1)
    f2.plot(wxx[integrals<1.01],integrals[integrals<1.01],'ro',color='green', label=r'major axis: d={:3.1f}mm'.format(diamy, prec='.1'))
    f2.plot(wyy[integrals<1.01],integrals[integrals<1.01],'ro',color='red',   label=r'minor axis: d={:3.1f}mm'.format(diamx, prec='.1'))
    f2.legend(loc="best", prop={'size':12})
    f2.set_xlim([lox,upx])
    f2.set_ylim([-0.01,1.01])    
    f2.set_xlabel('encircled radius (mm)',fontsize=13)
    f2.set_ylabel('intensity coverage',fontsize=13)
    f2.grid()
    
    #f2.hlines(beamdiameterdef,0.,upx, linestyles='dashed')

    g.tight_layout()

    g.savefig(outfile+"_majorminor.pdf")

    g3,f3 = plt.subplots(1,1)

    rx1 = 400 #rangey1
    rx2 = 1100 #rangey2
    ry1 = 300 #rangex1
    ry2 = 650 #rangex2
    
    #f3.imshow(scidata_sl2, origin='lower', cmap='gnuplot')
    f3.imshow(scidata_sl2[ry1:ry2,rx1:rx2], origin='lower', cmap='gnuplot')    

    xgr = np.arange(rx1,rx2,1) # the grid to which your data corresponds
    nx = xgr.shape[0]
    no_labels = 8 # how many labels to see on axis x
    step_x = int(nx / (no_labels - 1)) # step between consecutive labels
    x_positions = np.arange(0,nx,step_x) # pixel count at label position
    x_labels = roundto(xgr[::step_x]*conv,2) # labels you want to see
    f3.set_xticks(x_positions, x_labels)
    f3.set_xlabel('mm')    
    
    ygr = np.arange(ry1,ry2,1) # the grid to which your data corresponds
    ny = ygr.shape[0]
    no_labels = 8 # how many labels to see on axis x
    step_y = int(ny / (no_labels - 1)) # step between consecutive labels
    y_positions = np.arange(0,ny,step_y) # pixel count at label position
    y_labels = roundto(ygr[::step_y]*conv,2) # labels you want to see
    f3.set_yticks(y_positions, y_labels)
    f3.set_ylabel('mm')
    
    #f3.set(xticks=np.arange(0,(r2-r1)-1, 5), xticklabels=np.arange(0, conv*(r2-r1), conv*(r2-r1)/10));

    params_shifted = (height, x-ry1, y-rx1, width_x, width_y, theta, 0.)
    fit2 = gaussian(*params_shifted)
    
    f3.contour(fit2(*np.indices(scidata_sl2[ry1:ry2,rx1:rx2].shape))) #, cmap=plt.cm.copper)
    f3.text(rx1,ry1*0.9, r'$\sigma_x$={:3.1f}, $\sigma_y$={:3.1f}, $\alpha$={:.0f}$^\circ$'.format(width_y, width_x, theta*57.3,prec='.1'),
                     verticalalignment='bottom', horizontalalignment='left',
                     transform=axarr[0, 0].transAxes,
                     color='white', fontsize=8)       
    g3.savefig(outfile+"_img.pdf")


    
    idx2 = idx
    idy2 = idy
    if (idy > idx):
        idx2 = idy
    if (idx > idy):
        idy2 = idx

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    
    scidata = hdu.data
    scidata_sl = scidata[rangey1:rangey2,rangex1:rangex2]

    X = np.arange(idx2)  
    Y = np.arange(idy2)  
    XX, YY = np.meshgrid(X, Y)
    ZZ = zeros((idx2, idy2), dtype=float)
    for i in range(idx):
        for j in range(idy):
            ZZ[i, j] = scidata_sl[j][i]
    for i in range(idx2):
        for j in range(idy2):
            if (ZZ[i, j] == 0):
                ZZ[i, j] = ZZ[idx-1, idy-1]

    surf1 = ax1.plot_surface(XX, YY, ZZ, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    ax1.set_title(titles[0], color=colors[0])

    ax2 = fig.add_subplot(122, projection='3d')

    XL = arange(0, idx2, 1)  ######## this is Y (50)
    YL = arange(0, idy2, 1)  ######## this is X (45)
    XXL, YYL = np.meshgrid(XL, YL)
    ZZL = zeros((idx2, idy2), dtype=float)
    for i in range(idx):
        for j in range(idy):
            ZZL[i, j] = np.log(scidata_sl[j][i])
    for i in range(idx2):
        for j in range(idy2):
            if (ZZL[i, j] == 0):
                ZZL[i, j] = ZZL[idx-1, idy-1]

    surf2 = ax2.plot_surface(XXL, YYL, ZZL, linewidth=0, antialiased=False, cmap=cm.coolwarm)
    ax2.set_title(titles[1], color=colors[1])

    plt.show()

    fig.savefig(outfile+"_3.pdf")


if __name__ == '__main__':
    if len(sys.argv) > 1:   # there is at least one argument
        print('Number of arguments:', len(sys.argv), 'arguments.')
        print('Argument List:', str(sys.argv))

        file = sys.argv[1]
        rangey1 = int(float(sys.argv[2]))
        rangey2 = int(float(sys.argv[3]))
        rangex1 = int(float(sys.argv[4]))
        rangex2 = int(float(sys.argv[5]))
        #conv = float(sys.argv[6])

        run_script(file,rangey1,rangey2,rangex1,rangex2)


