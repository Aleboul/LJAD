import math
from scipy.integrate import quad, dblquad
import numpy as np
import matplotlib.pyplot as plt
import extreme_value_copula
import gumbel
from scipy import integrate
plt.style.use('seaborn-whitegrid')
def min(a, b):
      
    if a <= b:
        return a
    else:
        return b

#def C(x,y, lmbd, theta):
#    value_1 = x*y
#    value_2 = (1-x)*(1-y)
#    value_2 = 1 + theta*value_2
#    value_  = value_1 * value_2
#    return(value_)
#
#def C_1(x,y,lmbd,theta):
#    value_1 = y
#    value_2 = (1-2*x)*(1-y)
#    value_2 = 1 + theta*value_2
#    value_  = value_1 * value_2
#    return(value_)
#
#def C_2(x,y,lmbd,theta):
#    value_1 = x
#    value_2 = (1-2*y)*(1-x)
#    value_2 = 1 + theta*value_2
#    value_  = value_1 * value_2
#    return(value_)

def A(t, theta):
    value_ = math.pow(t, theta) + math.pow(1-t, theta)
    value_ = math.pow(value_,1/theta)
    return(value_)

def Aprime(t, theta):
    value_1 = math.pow(t, theta-1) - math.pow(1-t, theta-1)
    value_2 = math.pow(t, theta) + math.pow(1-t, theta)
    value_  = value_1 * math.pow(value_2 , (1/theta)-1)
    return(value_)

def kappa(t, theta):
    value_ = A(t, theta) - t*Aprime(t, theta)
    return(value_)

def zeta(t, theta):
    value_ = A(t, theta) + (1-t)*Aprime(t,theta)
    return(value_)

def C(x,y,lmbd, theta):
    value_1 = x*y
    value_2 = A(math.log(y) / math.log(x*y), theta)
    value_  = math.pow(value_1, value_2)
    return(value_)

def C_1(x,y,lmbd, theta):
    value_1 = C(x,y,lmbd, theta) / x
    value_2 = kappa(math.log(y)/math.log(x*y), theta)
    value_  = value_1 * value_2
    return(value_)

def C_2(x,y,lmbd, theta):
    value_1 = C(x,y,lmbd, theta) / y
    value_2 = zeta(math.log(y) / math.log(x*y), theta)
    value_  = value_1 * value_2
    return(value_)

def integrand_v1(x,y,lmbd, theta):
    u1 = math.pow(x,1/lmbd)
    u2 = math.pow(x,1/(1-lmbd))
    v1 = math.pow(y,1/(lmbd))
    v2 = math.pow(y,1/(1-lmbd))
    value_1 = C(min(u1,v1), min(u2,v2), lmbd, theta)
    value_2 = C(u1,u2, lmbd, theta)
    value_3 = C(v1,v2, lmbd, theta)

    value_  = value_1 - value_2 * value_3
    return(value_)

def integrand_v2(x,y,lmbd,theta):
    u1 = math.pow(x,1/lmbd)
    u2 = math.pow(x,1/(1-lmbd))
    v1 = math.pow(y,1/(lmbd))
    v2 = math.pow(y,1/(1-lmbd))
    value_1 = C_1(u1,u2,lmbd,theta)
    value_2 = C_1(v1,v2,lmbd, theta)
    value_3 = (min(u1, v1) - u1*v1)
    value_  = value_1 * value_2 * value_3
    return(value_)

def integrand_v3(x,y,lmbd,theta):
    u1 = math.pow(x,1/lmbd)
    u2 = math.pow(x,1/(1-lmbd))
    v1 = math.pow(y,1/(lmbd))
    v2 = math.pow(y,1/(1-lmbd))
    value_1 = C_2(u1,u2,lmbd,theta)
    value_2 = C_2(v1,v2,lmbd,theta)
    value_3 = (min(u2, v2) - u2*v2)
    value_  = value_1 * value_2 * value_3
    return(value_)

def integrand_cv12(x,y,lmbd,theta):
    u1 = math.pow(x,1/lmbd)
    u2 = math.pow(x,1/(1-lmbd))
    v1 = math.pow(y,1/(lmbd))
    v2 = math.pow(y,1/(1-lmbd))
    value_1 = C_1(v1,v2,lmbd,theta)
    value_2 = C(min(u1,v1), u2, lmbd,theta) - C(u1,u2,lmbd,theta) * v1
    value_  = value_1 * value_2
    return(value_)

def integrand_cv13(x,y,lmbd,theta):
    u1 = math.pow(x,1/lmbd)
    u2 = math.pow(x,1/(1-lmbd))
    v1 = math.pow(y,1/(lmbd))
    v2 = math.pow(y,1/(1-lmbd))
    value_1 = C_2(v1,v2,lmbd,theta)
    value_2 = C(u1, min(u2,v2), lmbd,theta) - v2 * C(u1,u2,lmbd,theta)
    value_  = value_1 * value_2
    return(value_)

def integrand_cv23(x,y,lmbd,theta):
    u1 = math.pow(x,1/lmbd)
    u2 = math.pow(x,1/(1-lmbd))
    v1 = math.pow(y,1/(lmbd))
    v2 = math.pow(y,1/(1-lmbd))
    value_1 = C_1(u1,u2,lmbd,theta)
    value_2 = C_2(v1,v2,lmbd,theta)
    value_3 = C(u1, v2, lmbd,theta) - u1 * v2
    value_  = value_1 * value_2 * value_3
    return(value_)

def var_FMado(lmbd, theta):
    v1 = dblquad(lambda x,y : integrand_v1(x,y, lmbd, theta), 0,1.0, 0, 1.0)[0]
    v2 = dblquad(lambda x,y : integrand_v2(x,y, lmbd, theta), 0,1.0, 0, 1.0)[0]
    v3 = dblquad(lambda x,y : integrand_v3(x,y, lmbd, theta), 0,1.0, 0, 1.0)[0]
    cv12 = dblquad(lambda x,y : integrand_cv12(x,y, lmbd, theta), 0,1.0, 0, 1.0)[0]
    cv13 = dblquad(lambda x,y : integrand_cv13(x,y, lmbd, theta), 0,1.0, 0, 1.0)[0]
    cv23 = dblquad(lambda x,y : integrand_cv23(x,y, lmbd, theta), 0,1.0, 0, 1.0)[0]

    return(v1 + v2 + v3 - 2*cv12 - 2*cv13 + 2* cv23)

def integrand_half_cv121(x,y,lmbd,theta):
    u1 = math.pow(x,1/lmbd)
    u2 = math.pow(x,1/(1-lmbd))
    v1 = math.pow(y,1/(lmbd))
    v2 = math.pow(y,1/(1-lmbd))
    value_1 = C(u1,u2, lmbd, theta)*(1-v1)*C_1(v1,v2, lmbd, theta)
    return(value_1)

def bounds_y_cv121():
    return [0.0,1.0]

def bounds_x_cv121(y):
    return[0.0, y]

def integrand_half_cv122(x,y,lmbd, theta):
    u1 = math.pow(x,1/lmbd)
    u2 = math.pow(x,1/(1-lmbd))
    v1 = math.pow(y,1/(lmbd))
    v2 = math.pow(y,1/(1-lmbd))
    value_1 = min(v1,u2) * C_1(v1,v2,lmbd,theta)
    value_2 = C(u1,u2,lmbd,theta)*v1 * C_1(v1,v2,lmbd,theta)
    return(value_1 - value_2)

def bounds_y_cv122():
    return [0.0,1]

def bounds_x_cv122(y):
    return[y, 1.0]

def integrand_half_cv131(x,y,lmbd,theta):
    u1 = math.pow(x,1/lmbd)
    u2 = math.pow(x,1/(1-lmbd))
    v1 = math.pow(y,1/(lmbd))
    v2 = math.pow(y,1/(1-lmbd))
    value_1 = C(u1,u2, lmbd, theta)*(1-v2)*C_2(v1,v2, lmbd, theta)
    return(value_1)


def integrand_half_cv132(x,y,lmbd, theta):
    u1 = math.pow(x,1/lmbd)
    u2 = math.pow(x,1/(1-lmbd))
    v1 = math.pow(y,1/(lmbd))
    v2 = math.pow(y,1/(1-lmbd))
    value_1 = min(u1,v2)*C_2(v1,v2,lmbd,theta)
    value_2 = C(u1,u2,lmbd,theta)*v2 * C_2(v1,v2,lmbd,theta)
    return(value_1-value_2)

def var_FMado(lmbd, theta):
    v1 = dblquad(lambda x,y : integrand_v1(x,y, lmbd, theta), 0,1.0, 0, 1.0)[0]
    v2 = dblquad(lambda x,y : integrand_v2(x,y, lmbd, theta), 0,1.0, 0, 1.0)[0]
    v3 = dblquad(lambda x,y : integrand_v3(x,y, lmbd, theta), 0,1.0, 0, 1.0)[0]
    cv12 = integrate.nquad(lambda x,y : integrand_half_cv122(x,y, lmbd, theta),[bounds_x_cv122, bounds_y_cv122])[0] + integrate.nquad(lambda x,y : integrand_half_cv121(x,y, lmbd, theta),[bounds_x_cv121, bounds_y_cv121])[0]
    cv13 = integrate.nquad(lambda x,y : integrand_half_cv132(x,y, lmbd, theta),[bounds_x_cv122, bounds_y_cv122])[0] + integrate.nquad(lambda x,y : integrand_half_cv131(x,y, lmbd, theta),[bounds_x_cv121, bounds_y_cv121])[0]
    print(v3)
    return(v1 + v2 + v3 - 2*cv12 - 2*cv13)

theta = 5
lmbd = 0.2

print(var_FMado(lmbd, theta))

#n = 50
#n_iter = 1000
#n_sample = [64]
#theta = 15
#random_seed = 42
#
#
#copula = gumbel.Gumbel(copula_type = 'GUMBEL', random_seed = 42, theta = theta, n_sample = np.max(n_sample))
#Monte = extreme_value_copula.Monte_Carlo(n_iter= n_iter, n_sample= n_sample, random_seed= random_seed, copula= copula)
#var_lmbd = Monte.exec_varlmbd(n, plot = False)
#
#x = np.linspace(0.01,0.99,n)
#value = []
#
#for lmbd in x:
#    value_ = var_FMado(lmbd, theta) 
#    value.append(value_)
#
#print(x)
#print(value)
#
#fig, ax = plt.subplots()
#ax.plot(x, value, '--', color = 'darkblue')
#ax.plot(x, var_lmbd, '.', markersize = 5, alpha = 0.5, color = 'salmon')
#ax.set_xlabel(r'$\sigma^2$')
#ax.set_ylabel(r'$\lambda$')
#plt.savefig("/home/aboulin/Documents/stage/naveau_2009/output/image_2.png")
