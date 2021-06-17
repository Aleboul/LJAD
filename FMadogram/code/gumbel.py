import numpy as np
from enum import Enum
from scipy.optimize import minimize_scalar
class CopulaTypes(Enum):
    """ Available copula families. """

    GUMBEL = 2
    INDEPENDENCE = 3

class Bivariate(object):
    """
        Base class for bivariate copulas.

        This class allows to instantiate all is subclasses and serves
        as a unique entry point for the bivariate copulas classes.

        Inputs
        ------
            copula_type : subtype of the copula
            random_seed : Seed for the random generator

        Attributes
        ----------
            copula_type(CopulaTypes) : Family of the copula a subclass belongs to
            theta_interval(list[float]) : Interval of vlid thetas for the given copula family
            invalid_thetas(list[float]) : Values that, even though they belong to
                :attr:`theta_interval`, shouldn't be considered valid.
            theta(float) : Parameter for the copula
    """
    copula_type = None
    _subclasses = []
    theta_interval = []
    invalid_thetas = []
    theta = []
    n_sample = []

    def __init__(self, copula_type = None, random_seed = None, theta = None, n_sample = None):
        """
            Initialize Bivariate object.

            Args:
            -----
                Copula_type (CopulaType or str) : subtype of the copula.
                random_seed (int or None) : Seed for the random generator
                theta (float or None) : Parameter for the copula.
        """
        self.random_seed = random_seed
        self.theta = theta
        self.n_sample = n_sample


class Gumbel(Bivariate):

    theta_interval = [0,1]
    invalid_thetas = []

    def generate_randomness(self):
        """
            Generate a bivariate sample draw identically and
            independently from a uniform over the segment [0,1]

            Inputs
            ------
            self.n_sample : length of the bivariate sample

            Outputs
            -------
            n_sample x 2 np.array
        """

        v_1 = np.random.uniform(low = 0.0, high = 1.0, size = self.n_sample) # first sample
        v_2 = np.random.uniform(low = 0.0, high = 1.0, size = self.n_sample) # second sample
        output_ = np.vstack([v_1, v_2]).T
        return output_

    def generator(self, t):
        """
            Return the generator function

            Inputs
            ------
            t : value of the generator
            self.theta : parameter of the copula fonction

            Outputs
            -------
            real number
        """
        value_ = np.power((-np.log(t)),self.theta)
        return(value_)
    
    def K_c(self, t):
        """
            Return the derivative of the generator function
        """
        value_ = t*(1-(np.log(t)/self.theta))
        return value_

    def sample(self):
        output = np.zeros((self.n_sample,2))
        X = self.generate_randomness()
        for i in range(0,self.n_sample):
            v = X[i]
            def func(x):
                value_ = np.abs(self.K_c(x) - v[1])
                return(value_)
            sol = minimize_scalar(func, bounds = (0,1), method = "bounded")
            sol = float(sol.x)
            u = [np.exp(np.power(v[0],1/self.theta)*np.log(sol)) , np.exp(np.power(1-v[0],1/self.theta)*np.log(sol))]
            output[i,:] = u
        return output

copula = Gumbel(copula_type= "GUMBEL", random_seed= 42, theta=2, n_sample = 1)
copula.sample()