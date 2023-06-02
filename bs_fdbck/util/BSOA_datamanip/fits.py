import numpy as np
import scipy
import scipy.optimize
from scipy import odr
import matplotlib.pyplot as plt

from bs_fdbck.util.plot.BSOA_plots import cdic_model

def func_exp(x, a, b):
    return a * np.exp(b * x)

def target_function_exp_fit(p, x):
    a,b= p
    return a*np.exp(b*x)

def func_exp_wc(x, a, b, c):
    return a * np.exp(b * x) + c

def target_function_exp_fit_wc(p, x):
    a,b,c = p
    return a*np.exp(b*x)+c

def get_exp_fit(df_s, v_x, v_y, func=func_exp, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values

    popt, pcov = scipy.optimize.curve_fit(func, x, y)
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    label = '$%5.2f e^{%5.2fx}$' % tuple(popt)
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label

    # plt.plot(x, func_exp(x, *popt), 'r-',
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))


def get_exp_fit_weight(df_s, v_x, v_y, func=func_exp, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values

    popt, pcov = scipy.optimize.curve_fit(func, x, y, sigma=np.log10(1 + y), absolute_sigma=False)
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    label = exp_lab(popt)
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label


def exp_lab(popt):
    a = popt[0]
    b = popt[1]
    if np.abs(a)< 0.009:
        #a_lab = ((str("%.2e" % a)).replace("e", ' \\cdot 10^{ ')).replace("+0", ") + ' } ')
        label = '($%.1E) \cdot e^{%5.2fx}$' % tuple(popt)
    else:
        label = '$%5.2f e^{%5.2fx}$' % tuple(popt)

    return label


def get_exp_fit_wc(df_s, v_x, v_y, func=func_exp_wc, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values

    popt, pcov = scipy.optimize.curve_fit(func, x, y)
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

    label = exp_wc_lab( popt)
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label

    # plt.plot(x, func_exp(x, *popt), 'r-',
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))


def exp_wc_lab( popt):
    cs = '%5.2f' % popt[-1]
    if cs[0] != '-':
        cs = '+' + cs
    label = '$%5.2f e^{%5.2fx}$' % tuple(popt[:-1])
    label = label + cs
    return label


def func_lin_fit(x, a, b):
    return a * x + b

def func_ax_fit(x, a,):
    return a * x


def func_lin_fit_r(x, *beta):
    return beta[0]*x + beta[1]
    #return a + x * b

def target_function_linear(p, x):
    m, c = p
    return m*x + c

def target_function_ax(p, x):
    m, = p
    return m*x


def get_linear_fit(df_s, v_x, v_y, func=func_lin_fit, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values

    popt, pcov = scipy.optimize.curve_fit(func, x, y)
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f' % popt[-1]
    label = '$%5.2fx+ %5.2f$' % tuple(popt)
    label = label
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label


def get_least_square_fit_and_labs(df_s, v_x, v_y, fit_func = 'linear', return_func=False,
                         return_out_obj = False,
                         least_square_kwrgs=None,
                         beta0=None):
    if fit_func =='linear':
        func = target_function_linear
        out_func = func_lin_fit
        lab_func = lin_lab
        if beta0 is None:
            beta0 = [0,0]
    if fit_func == 'ax' :
        func = target_function_ax
        out_func =  func_ax_fit
        lab_func = ax_lab
        if beta0 is None:
            beta0 = [0]
    elif fit_func =='exp':
        func = target_function_exp_fit
        out_func = func_exp
        lab_func = exp_lab
        if beta0 is None:

            beta0 = [0,0]
    elif fit_func == 'exp_wc':
        func = target_function_exp_fit_wc
        out_func = func_exp_wc
        lab_func = exp_wc_lab
        if beta0 is None:

            beta0 = [0,0,0]

    elif fit_func =='log':
        func = target_function_log
        out_func = func_log
        lab_func = log_lab
        if beta0 is None:
            beta0 = [0,0]
    elif fit_func =='log_abc':
        func = target_function_log_abc
        out_func = func_log_abc
        lab_func = log_abc_lab
        if beta0 is None:

            beta0 = [1092,385,-0.26]



    popt, pcov = get_least_squares_fit(df_s,
                                       out_func,
                                       v_x,v_y,
                                       beta0,
                                       least_square_kwrgs= least_square_kwrgs)
    print('Going for least square')
    label = lab_func(popt)
    if return_func:
        return popt, pcov, label, out_func

    return popt, pcov, label
def get_odr_fit_and_labs(df_s, v_x, v_y, fit_func = 'linear', return_func=False,
                         return_out_obj = False,
                         least_square_kwrgs=None,
                         pprint=True,
                         beta0=None):
    if fit_func =='linear':
        func = target_function_linear
        out_func = func_lin_fit
        lab_func = lin_lab
        #if beta0 is None:
         #   beta0 = [0,0]
    if fit_func == 'ax' :
        func = target_function_ax
        out_func =  func_ax_fit
        lab_func = ax_lab
        #if beta0 is None:
        #    beta0 = [0]
    elif fit_func =='exp':
        func = target_function_exp_fit
        out_func = func_exp
        lab_func = exp_lab
        #if beta0 is None:
        #    beta0 = [0,0]
    elif fit_func == 'exp_wc':
        func = target_function_exp_fit_wc
        out_func = func_exp_wc
        lab_func = exp_wc_lab
        #if beta0 is None:
        #    beta0 = [0,0,0]

    elif fit_func =='log':
        func = target_function_log
        out_func = func_log
        lab_func = log_lab
        #if beta0 is None:
        #    beta0 = [0,0]
    elif fit_func =='log_abc':
        func = target_function_log_abc
        out_func = func_log_abc
        lab_func = log_abc_lab
        #if beta0 is None:
        #    beta0 = [1092,385,-0.26]


    ls_popt, ls_pcov = get_least_squares_fit(df_s, out_func, v_x,v_y,beta0, least_square_kwrgs= least_square_kwrgs)
    if beta0 is None:
        beta0 = ls_popt

    print(beta0)

    # df_s_norm =(df_s-df_s.mean())/df_s.std()

    out = get_odr_fit(df_s, func, v_x, v_y, beta0=beta0, pprint=pprint)
    # out.pprint()
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f' % popt[-1]
    popt = out.beta
    print(popt)
    pcov = out.cov_beta

    label = lab_func(popt)
    label = label
    print('reason for haltng')

    if out.stopreason[0]=='Iteration limit reached':
        popt, pcov = get_least_squares_fit(df_s, out_func, v_x,v_y,beta0, least_square_kwrgs= least_square_kwrgs)
        print('Going for least square')
        label = lab_func(popt)
    if return_func:
        if return_out_obj:
            return popt, pcov, label, out_func, out
        else:
            return popt, pcov, label, out_func
    if return_out_obj:
        return popt, pcov, label, out

    return popt, pcov, label


def lin_lab(popt):
    label = '$%5.2fx+ %5.2f$' % tuple(popt)
    return label

def ax_lab(popt):
    label = '$%5.2fx$' % tuple(popt)
    return label

def get_odr_fit(df_s, func, v_x, v_y, beta0, pprint = True):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values
    odr_model = odr.Model(func)
    data = odr.RealData(x=x, y=y, sx=np.std(x), sy=np.std(y))
    ordinal_distance_reg = odr.ODR(data, odr_model,
                                   beta0=beta0,
                                   maxit=500,
                                   )

    # popt, pcov = odr(func, x, y)
    out = ordinal_distance_reg.run()
    if pprint:
        out.pprint()
    return out
def get_least_squares_fit(df_s, func, v_x, v_y, beta0,least_square_kwrgs=None):
    if least_square_kwrgs is None:
        least_square_kwrgs = dict()
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values
    popt, pcov = scipy.optimize.curve_fit(func, x, y, **least_square_kwrgs)

    #odr_model = odr.Model(func)
    #data = odr.Data(x, y)
    #ordinal_distance_reg = odr.ODR(data, odr_model,
    #                               beta0=beta0
    #                               )

    # popt, pcov = odr(func, x, y)
    #out = ordinal_distance_reg.run()
    # out.pprint()
    return popt, pcov

def get_linear_fit_odr(df_s, v_x, v_y, func=target_function_linear, return_func=False, beta0 = None):
    if beta0 is None:
        beta0 = [1,1]
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values
    odr_model = odr.Model(func)
    data = odr.Data(x, y)
    ordinal_distance_reg = odr.ODR(data, odr_model,
                               beta0=beta0)

    #popt, pcov = odr(func, x, y)
    out = ordinal_distance_reg.run()
    #out.pprint()
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f' % popt[-1]
    popt = out.beta
    print(popt)
    pcov = out.cov_beta
    label = '$%5.2fx+ %5.2f$' % tuple(popt)
    label = label

    if return_func:
        return popt, pcov, label, func_lin_fit_r
    return popt, pcov, label


def func_poly2_fit(x, a, b, c):
    return a * x ** 2 + b * x + c


def get_poly2_fit(df_s, v_x, v_y, func=func_poly2_fit, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values

    popt, pcov = scipy.optimize.curve_fit(func, x, y, p0=[0, 0, 0])
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f' % popt[-1]
    label = '$%5.2fx^2+ %5.2fx + %5.2f$' % tuple(popt)
    label = label
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label


def get_poly2_fit_weight(df_s, v_x, v_y, func=func_poly2_fit, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values
    # yma = np.max(y)
    # ymi = np.min(y) * 0.6

    sig = np.log10((1 + y))

    popt, pcov = scipy.optimize.curve_fit(func, x, y, p0=[0, 0, 0], sigma=sig)
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f' % popt[-1]
    label = '$%5.2fx^2+ %5.2fx + %5.2f$' % tuple(popt)
    label = label
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label

    # plt.plot(x, func_exp(x, *popt), 'r-',
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))


def func_xa_fit(x, a, b, c):
    return b * (x + c) ** a


def get_xa_fit(df_s, v_x, v_y, func=func_xa_fit, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values

    popt, pcov = scipy.optimize.curve_fit(func, x, y, )
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f' % popt[-1]

    label = '$%5.2f (x+%5.2f)^{%5.2f}$' % tuple(popt)

    label = label
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label


def get_xa_fit_weight(df_s, v_x, v_y, func=func_xa_fit, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values
    # yma = np.max(y)
    # ymi = np.min(y) * 0.6

    sig = np.log10((1 + x))

    popt, pcov = scipy.optimize.curve_fit(func, x, y, sigma=sig)
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f' % popt[-1]
    label = '$%5.2f (x+%5.2f)^{%5.2f}$, weighted' % tuple(popt)

    label = label
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label

    # plt.plot(x, func_exp(x, *popt), 'r-',
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))


def func_lin_fit_a0(x, a):
    return a * x


def get_linear_fit_a0(df_s, v_x, v_y, func=func_lin_fit_a0, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values

    popt, pcov = scipy.optimize.curve_fit(func, x, y)
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f'%popt[-1]
    label = '$%5.2fx$ ' % popt[0]
    label = label
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label

    # plt.plot(x, func_exp(x, *popt), 'r-',
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))


def func_log(x, a, b):
    return a + b * np.log(x)

def target_function_log(p, x):
    a,b = p
    return a*np.exp(b*x)

def get_log_fit(df_s, v_x, v_y, func=func_log,use_ord_log = False,  return_func=False):
    if use_ord_log:
        return
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values

    popt, pcov = scipy.optimize.curve_fit(func, x, y)
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f'%popt[-1]
    label = log_lab(popt)
    label = label
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label

    # plt.plot(x, func_exp(x, *popt), 'r-',
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))


def log_lab(popt):
    label = '$%5.2f+ %5.2f \ln(x)$' % tuple(popt)
    return label


def get_log_fit_weight(df_s, v_x, v_y, func=func_log, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values
    # yma = np.max(y)
    # ymi = np.min(y) * 0.6

    sig = np.log10((1 + x))

    popt, pcov = scipy.optimize.curve_fit(func, x, y, sigma=sig, absolute_sigma=False)
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f'%popt[-1]
    label = '$%5.2f+ %5.2f \ln(x)$' % tuple(popt)
    label = label
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label

    # plt.plot(x, func_exp(x, *popt), 'r-',
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))


def func_log_a0(x, b):
    return b * np.log(x)


def get_log_fit_a0(df_s, v_x, v_y, func=func_log_a0, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values

    popt, pcov = scipy.optimize.curve_fit(func, x, y)
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f'%popt[-1]
    label = '$ %5.2f \ln(x)$' % tuple(popt)
    label = label
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label

    # plt.plot(x, func_exp(x, *popt), 'r-',
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))


def func_log_abc(x, a, b, c):
    return a + b * np.log(c + x)

def target_function_log_abc(p, x):
    a,b,c = p
    return a + b * np.log(c + x)

def get_log_fit_abc(df_s, v_x, v_y, func=func_log_abc, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values

    popt, pcov = scipy.optimize.curve_fit(func, x, y,maxfev=10000)
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f'%popt[-1]
    label = log_abc_lab(popt)
    label = label
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label


def log_abc_lab(popt):
    label = '$ %5.2f + %5.2f\ln(%5.2f+x)$' % tuple(popt)
    return label


def get_log_fit_abc_weight(df_s, v_x, v_y, func=func_log_abc, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values
    sig = np.log10((1 + x))

    popt, pcov = scipy.optimize.curve_fit(func, x, y, sigma=sig, absolute_sigma=False)

    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f'%popt[-1]
    label = '$ %5.2f + %5.2f\ln(%5.2f+x)$' % tuple(popt)
    label = label
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label

    # plt.plot(x, func_exp(x, *popt), 'r-',
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))


def func_log_abcmult(x, a, b, c):
    return a + b * np.log(c * x)


def get_log_fit_abcmult(df_s, v_x, v_y, func=func_log_abcmult, return_func=False):
    _df = df_s[[v_x, v_y]].dropna()
    y = _df[v_y].values
    x = _df[v_x].values

    popt, pcov = scipy.optimize.curve_fit(func, x, y)
    # lab = 'fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    # cs = '%5.2f'%popt[-1]
    label = '$ %5.2f + %5.2f\ln(%5.2f*x)$' % tuple(popt)
    label = label
    # x = np.linspace(*xlims)
    if return_func:
        return popt, pcov, label, func
    return popt, pcov, label

    # plt.plot(x, func_exp(x, *popt), 'r-',
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

def plot_fit(func, popt, mo, xlims, yscale, xscale, ax,label,extra_plot=False, **kwrgs):
    x = np.linspace(*xlims)
    if not extra_plot:
        ax.plot(x, func(x, *popt), c='w', linewidth=3,label='__nolegend__', )
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax.plot(x, func(x, *popt), linewidth=2, c=cdic_model[mo],label=f'{label}',**kwrgs)
    #     label='fit: %5.3f exp( %5.3f x) +  %5.3f' % tuple(popt))

    ax.set_yscale(yscale)
    ax.set_xscale(xscale)

