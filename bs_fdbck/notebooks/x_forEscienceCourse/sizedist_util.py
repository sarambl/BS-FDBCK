import xarray as xr
import numpy as np


import numpy as np
from scipy.stats import lognorm



def _compute_dNdlogD_mod(diameter,
                         input_ds,
                         mode_num,
                         ):
    # %%

    varN = f'NCONC{mode_num:02d}'
    varNMR = f'NMR{mode_num:02d}'
    varNMD = f'NMD{mode_num:02d}'
    varSIG = f'SIGMA{mode_num:02d}'
    dNdlogD_var = f'dN(mode {mode_num:02d})/dlogD'
    input_ds[varNMD] = 2.*input_ds[varNMR]
    #print(varN)
    # %%
    size_dtset = xr.Dataset(coords={**input_ds.coords, 'diameter': diameter})
    # varNMR = varListNorESM['NMR'][i]
    NCONC = input_ds[varN]  # [::]*10**(-6) #m-3 --> cm-3
    SIGMA = input_ds[varSIG]  # [::]#*10**6
    NMD = input_ds[varNMD]   # radius --> diameter
    # number:
    size_dtset[dNdlogD_var] = dNdlogD_modal(NCONC, NMD, SIGMA, diameter)
    size_dtset[dNdlogD_var].attrs['units'] = 'cm-3'
    size_dtset[dNdlogD_var].attrs['long_name'] = 'dN/dlogD (mode' + dNdlogD_var[-2:] + ')'
    return size_dtset


def dNdlogD_sec(diameter, SOA, SO4, num, bin_diameter_int):
    """
    Calculate dNdlogD sectional for individual bin
    :param diameter:
    :param SOA:
    :param SO4:
    :param num:
    :param bin_diameter_int:
    :return:
    """
    # SECnr = self.nr_bins
    # binDiam_l = self.bin_diameter_int  # binDiameter_l
    if type(num) is str:
        num = int(num)

    diam_u = bin_diameter_int[num]  # bin upper lim
    diam_l = bin_diameter_int[num - 1]  # bin lower lim

    SOA, dum = xr.broadcast(SOA, diameter)  # .values  # *1e-6
    SO4, dum = xr.broadcast(SO4, diameter)  # .values  # *1e-6
    dNdlogD = (SOA + SO4) / (np.log(diam_u / diam_l))  #
    in_xr = dict(diameter=diameter[(diameter >= diam_l) & (diameter < diam_u)])
    out_da = xr.DataArray(np.zeros_like(dNdlogD.values), coords=dNdlogD.coords)  # dNdlogD*0.#, diameter)
    out_da.loc[in_xr] = dNdlogD.loc[in_xr]
    return out_da


def get_nrSEC_varname(num):
    """
    Get necessary input nr sec variables
    :param num: number e.g. 02
    :return: list of variables.
    """
    SOA = 'nrSOA_SEC'
    SO4 = 'nrSO4_SEC'
    if type(num) is int:
        num = '%02.0f' % num
    return [SOA + num, SO4 + num]
    # fl = ['nrSOA%s']


def dNdlogD_modal(NCONC, NMD, SIGMA, diameter):
    """

    :param NCONC:
    :param NMD: in diameter!!
    :param SIGMA:
    :param diameter:
    :return:
    """
    da = NCONC / (np.log10(SIGMA) * np.sqrt(2 * np.pi)) * np.exp(
        -(np.log10(diameter) - np.log10(NMD)) ** 2 / (2 * np.log10(SIGMA) ** 2))
    return da


def _varname_sec_nr(varN):
    dNdlogD_var = 'dNdlogD_sec%s' % (varN[-2::])
    return dNdlogD_var


def _varname_mod_nr(varN):
    dNdlogD_var = 'dNdlogD_mode%s' % (varN[-2::])
    return dNdlogD_var


def _get_sig_varname(num):
    return 'SIGMA%s' % num


def _get_nmr_varname(num):
    return 'NMR%s' % num


def _get_nconc_varname(num):
    return 'NCONC%s' % num



def lognormal_julia(x, N, mu, sigma):
    '''Function that defines the lognormal distribution.

    Parameters:
        x        :   (np.array) The particle diameters (in micrometres) at which you wan to evaluate dN/dlogD
        N        :   The number concentration in this particular aerosol mode (MODEL OUTPUT)
        mu       :   The mean modal radius (NB!) (MODEL OUTPUT)
        sigma :   The  standard deviation of this mode (MODEL OUTPUT)

    Returns an array of the same size as x, with dNdlogD values.'''
    logsigma = np.log10(sigma)
    return N * (1/np.sqrt(2*np.pi)) * (1/logsigma) * np.exp(-np.log10(x/(2*mu))**2 / (2 * logsigma**2))




sized_varListNorESM = {'NCONC': ['NCONC00', 'NCONC01', 'NCONC02', 'NCONC04', 'NCONC05', 'NCONC06', 'NCONC07', 'NCONC08',
                           'NCONC09', 'NCONC10', 'NCONC12', 'NCONC14'],
                 'SIGMA': ['SIGMA00','SIGMA01', 'SIGMA02', 'SIGMA04', 'SIGMA05', 'SIGMA06', 'SIGMA07', 'SIGMA08',
                           'SIGMA09', 'SIGMA10', 'SIGMA12', 'SIGMA14'],
                 'NMR': ['NMR00','NMR01', 'NMR02', 'NMR04', 'NMR05', 'NMR06', 'NMR07', 'NMR08',
                         'NMR09', 'NMR10', 'NMR12', 'NMR14']}


sized_varlist_SOA_SEC = ['nrSOA_SEC01', 'nrSOA_SEC02', 'nrSOA_SEC03', 'nrSOA_SEC04', 'nrSOA_SEC05']
sized_varlist_SO4_SEC = ['nrSO4_SEC01', 'nrSO4_SEC02', 'nrSO4_SEC03', 'nrSO4_SEC04', 'nrSO4_SEC05']
list_sized_vars_noresm = sized_varListNorESM['NCONC'] + \
                         sized_varListNorESM['SIGMA'] + \
                         sized_varListNorESM['NMR'] + \
                         sized_varlist_SOA_SEC + \
                         sized_varlist_SO4_SEC
list_sized_vars_nonsec = sized_varListNorESM['NCONC'] + sized_varListNorESM['SIGMA'] + sized_varListNorESM['NMR']




def calc_Nd_interval_NorESM(input_ds, fromNd, toNd, varNameN):
    varN = 'NCONC%02.0f' % 1
    da_Nd = input_ds[varN] * 0.  # keep dimensions, zero value
    da_Nd.name = varNameN
    da_Nd.attrs['long_name'] ='N$_{%.0f-%.0f}$'%(fromNd,toNd)
    varsNCONC = sized_varListNorESM['NCONC'][0:1]
    varsNMR = sized_varListNorESM['NMR'][0:1] #radius --> diameter
    varsSIG = sized_varListNorESM['SIGMA'][0:1]
    for varN, varSIG, varNMR in zip(varsNCONC, varsSIG, varsNMR):
        print(varN)
        NCONC = input_ds[varN].values  # *10**(-6) #m-3 --> cm-3
        SIGMA = input_ds[varSIG].values  # case[varSIG][lev]#*10**6
        NMR = input_ds[varNMR].values * 2  #  radius --> diameter

        # nconc_ab_nlim[case][model]+=logR*NCONC*lognorm.pdf(logR, np.log(SIGMA),scale=NMR)
        if fromNd > 0:
            dummy = NCONC * (lognorm.cdf(toNd, np.log(SIGMA), scale=NMR)) - NCONC * (
                    lognorm.cdf(fromNd, np.log(SIGMA), scale=NMR))
        else:
            dummy = NCONC * (lognorm.cdf(toNd, np.log(SIGMA), scale=NMR))
        # if NMR=0 --> nan values. We set these to zero:
        dummy[NMR == 0] = 0.
        dummy[NCONC == 0] = 0.
        dummy[np.isnan(NCONC)] = np.nan
        da_Nd+= dummy
    return da_Nd

def get_N_nice_name_Nd(N_name):
    splt=N_name.split('_')
    if len(splt)==2:
        N_fancy_name='N$_{d<%s}$'%splt[1]
    else:
        N_fancy_name= 'N$_{%s<d<%s}$'%(splt[0][1::],splt[2])
    return N_fancy_name

