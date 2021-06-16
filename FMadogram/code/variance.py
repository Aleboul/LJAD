import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib import cm
from tqdm import tqdm

def A(t, theta):
    value_ = np.power(t, 1/theta) + np.power(1-t, 1/theta)
    value_ = np.power(value_, theta)
    return(value_)

def Aprime(t, theta):
    value_1 = np.power(t, (1/theta) - 1) / theta - np.power(1-t, (1/theta) - 1) / theta
    value_2 = np.power(t, 1/theta) + np.power(1-t,1/theta)
    value_ = theta * (value_1) * np.power(value_2, theta - 1)
    return(value_)

#def A(t, theta):
#    value_ = np.power(t, -1/theta) + np.power(1-t, -1/theta)
#    value_ = np.power(value_, -theta)
#    return(1-value_)
#
#def Aprime(t, theta):
#    value_1 = - np.power(t, -(1/theta) - 1) / theta + np.power(1-t, -(1/theta) - 1) / theta
#    value_2 = np.power(1/t, 1/theta) + np.power(1/(1-t),1/theta)
#    value_ = theta * (value_1) * np.power(value_2, -theta - 1)
#    return(value_)

def kappa(lmbd, A):
    value_ = A(1-lmbd, theta) + (1-lmbd)*Aprime(1-lmbd, theta)
    return(value_)

def zeta(lmbd, A):
    value_ = A(1-lmbd, theta) - (1-lmbd)*Aprime(1-lmbd, theta)
    return(value_)

def f(lmbd, A):
    value_ = np.power(lmbd*(1-lmbd)/(A(1-lmbd, theta) + lmbd*(1-lmbd)),2)
    return(value_)

def f1(lmbd, A, kappa, x):
    value_1 = x * np.power(1-x,3) * kappa(lmbd, A)
    value_2 = (A(1-lmbd, theta) + lmbd*(1-lmbd))*(A(1-lmbd, theta) + (1-x))
    value_ = value_1 / value_2
    return(value_)

def f2(lmbd, A, kappa, x):
    value_1 = x*(1-x)*kappa(lmbd, A)
    value_2 = A(1-lmbd, theta) - (1-x) + lmbd*(1-lmbd)
    value_3 = x*(1-x) / (A(1-lmbd, theta) + x - (1-x) + 2*lmbd*(1-lmbd))
    value_4 = np.power(1-x,2) / (A(1-lmbd, theta) + 2*lmbd*(1-lmbd))
    value_  = (value_1 / value_2) * (value_3 - value_4)
    return(value_)

def f_kappa(lmbd,A):
    if lmbd > 0.5 :
        value_ = f1(lmbd, A, kappa, lmbd) + f2(lmbd, A, kappa, lmbd)
        return(value_)
    else:
        value_1 = kappa(lmbd, A) * np.power(lmbd*(1-lmbd), 2)
        value_2 = (A(1-lmbd, theta) + lmbd*(1-lmbd))*(A(1-lmbd, theta) + 2*lmbd*(1-lmbd))
        value_ = value_1 / value_2
        return(value_)

def f_zeta(lmbd, A):
    if lmbd > 0.5 :
        value_1 = zeta(lmbd, A) * np.power(lmbd*(1-lmbd),2)
        value_2 = (A(1-lmbd, theta) + lmbd*(1-lmbd))*(A(1-lmbd, theta) + 2*lmbd*(1-lmbd))
        value_ = value_1 / value_2
        return(value_)
    else :
        value_ = f1(lmbd, A, zeta, 1-lmbd) + f2(lmbd, A, zeta, 1-lmbd)
        return(value_)

def common(lmbd, A):
    value_1 = A(1-lmbd, theta) / (A(1-lmbd, theta) + 2*lmbd*(1-lmbd))
    value_2 = np.power(kappa(lmbd, A),2) * (1-lmbd) / (2*A(1-lmbd, theta) - (1-lmbd) + 2*lmbd*(1-lmbd))
    value_3 = np.power(zeta(lmbd, A),2) * lmbd / (2*A(1-lmbd, theta) - lmbd + 2*lmbd*(1-lmbd))
    return f(lmbd, A)* (value_1 + value_2 + value_3)

def lower(lmbd, A):
    ### second term
    value_1 = (np.power(1-lmbd,2) - A(1-lmbd, theta)) / (2*A(1-lmbd, theta) - (1-lmbd) + 2*lmbd*(1-lmbd))
    value_1 = 2*kappa(lmbd, A) * f(lmbd,A) * (value_1) + 2*f_kappa(lmbd, A)
    ### third term
    value_2 = (np.power(lmbd,2) - A(1-lmbd, theta)) / (2*A(1-lmbd,theta) - lmbd + 2*lmbd*(1-lmbd))
    value_2 = 2*zeta(lmbd, A) * f(lmbd, A) * value_2 + 2 *f_zeta(lmbd, A)
    ### result
    value_ = common(lmbd, A) - value_1 - value_2
    return(np.maximum(value_,0))

def upper(lmbd, A):
    ### second term
    value_1 = (1-lmbd) / (2*A(1-lmbd, theta) - (1-lmbd) + 2 * lmbd*(1-lmbd)) + (A(1-lmbd, theta) - lmbd) / (A(1-lmbd, theta) + lmbd + 2*lmbd*(1-lmbd))
    value_1 = kappa(lmbd, A) * f(lmbd, A) * value_1
    ### third term
    value_2 = lmbd / (2*A(1-lmbd, theta) - lmbd + 2*lmbd*(1-lmbd)) + (A(1-lmbd, theta) - (1-lmbd)) / (A(1-lmbd, theta) + 1 - lmbd + 2*lmbd*(1-lmbd))
    value_2 = zeta(lmbd, A) * f(lmbd, A) * value_2
    ### fourth term
    value_3 = (zeta(lmbd, A) * kappa(lmbd, A) * lmbd*(1-lmbd)) / A(1-lmbd, theta)
    value_3 = 2 * f(lmbd, A) * value_3
    ### result
    value_ = common(lmbd, A) - value_1 - value_2  + value_3
    return(value_)

def var_mado_missing(x, p_xy, p_x, p_y):
	value = ((x ** 2 * (1-x)**2) / (1+x*(1-x))**2) * ( (p_xy**-1) / (1+2*x*(1-x)) - (p_x**-1)* (1-x) / (1+x+2*x*(1-x)) - (p_y**-1)*x / (2-x+2*x*(1-x)))
	return value
theta = 0.15

x = np.linspace(0.0,1.0,100)
values_upper = [upper(lmbd, A) for lmbd in x]
values_FMado = [var_mado_missing(lmbd,1,1,1) for lmbd in x]
values_lower = [lower(lmbd,A) for lmbd in x]

print(values_lower)

fig, ax = plt.subplots()

plt.plot(x, values_lower)
plt.plot(x, values_upper)
#plt.plot(x, values_FMado)
plt.savefig("/home/aboulin/Documents/stage/naveau_2009/output/variance.png")