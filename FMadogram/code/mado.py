import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib import cm
from tqdm import tqdm
import warnings

class simulation(object):
    """
        Base class for monte carlo simulation

        Inputs
        ------
            n_iter : number of Monte Carlo simulation
            simulation : join law of data
            n_sample : multiple length of sample
            lmbd : value of lmbd
            random_seed (Union[int, None]) : Seed for the random generator
        
        Attributes
        ----------
            n_sample(list[int]) : several lengths used for estimation
            lmbd(float) : Parameter for the lmbd-FMadogram
            lmbd_interval : Interval of valid thetas for the given copula
            var_mado : Theoretical variance computed with independent copula
            true_FMado : Theoretical value of the lmbd-FMadogram
    """

    n_iter = None
    simulation = None
    n_sample = []
    lmbd = None
    lmbd_interval = []

    def __init__(self, n_iter = None, simulation = None, n_sample = [], lmbd = None, random_seed = None):
        """
            Initialize simulation object.
        """

        self.n_iter = n_iter
        self.simulation = simulation
        self.n_sample = n_sample
        self.lmbd = lmbd

    def check_lmbd(self):
        """
            Validate the lmbd

            This method is used to assert the lmbd insert by the user.

            Raises :
                ValueError : If lmbd is not in :attr:`lmbd_interval`
        """

        lower, upper = self.lmbd_interval
        if (not lower <= sef.lmbd <= upper):
            message = "The lmbd value {} is out of limits for the given estimation."
            raise ValueError(message.format(self.lmbd))

    def _ecdf(self,data):
        """
            Compute ECDF of the data

            Inputs
            ------
            data : array of data

            Outputs
            -------
            Empirical cumulative distribution function
        """

        index = np.argsort(data)
        ecdf = np.zeros(len(index))
        for i in index:
            ecdf[i] = (1.0 / len(index)) * np.sum(data <= data[i])

        return ecdf
    
    def _dist(self, X):
        """
	    	Compute matrix of F-Madogram (a une constante pres) using the empirical cumulative distribution function
    
	    	Inputs
	    	------
	    	X : a matrix composed of ecdf
	    	lmbd : a parameter between 0 and 1

	    	Outputs
	    	-------
	    	A matrix with quantity equals to 0 if i=j (diagonal) and equals to sum_t=1^T |F(X_t)^{\lmbd} - G(Y_t)^{1-\lmbd}| if i \neq j
	    """

        ncols = X.shape[1]
        nrows = X.shape[0]
        dist = np.zeros([nrows, nrows]) # initialization
        for i in range(0, nrows):
            for j in range(0,i):
                if i==j :
                    dist[i,i] = 0
                else :
                    F_x = np.squeeze(X[j,:])
                    G_y = np.squeeze(X[i,:])
                    d = np.linalg.norm((np.power(F_x,self.lmbd) - np.power(G_y,1-self.lmbd)), ord = 1) - (self.lmbd) * np.sum((1-np.power(F_x,self.lmbd))) - (1-self.lmbd) * np.sum((1 - np.power(G_y, 1 - self.lmbd))) # formula of the normalized lmbd madogram estimator, see Naveau 2009
                    dist[i,j] = d
                    dist[j,i] = d

        return dist

    def _fmado(self, X):
        """
		    This function computes the lmbd FMadogram
    
		    Inputs
		    ------
		    X : a matrix
		    lmbd : constant between 0 and 1 use for the lmbd F-madogram

		    Outputs
		    -------
		    A matrix equals to 0 if i = j and equals to |F(X)^{\lmbd} - G(Y)^{1-\lmbd}|
	    """

        Nnb = X.shape[1]
        Tnb = X.shape[0]

        V = np.zeros([Tnb, Nnb])
        for p in range(0, Nnb):
            X_vec = np.array(X[:,p])
            Femp = self._ecdf(X_vec)
            V[:,p] = Femp
        Fmado = self._dist(np.transpose(V)) / (2 * Tnb) + (1/2) * ((1 - self.lmbd*(1-self.lmbd))/ ((2-self.lmbd)*(1+self.lmbd)))

        return Fmado

    def var_mado(self,x):
	    value = ((x ** 2 * (1-x)**2) / (1+x*(1-x))**2) * ( 1 / (1+2*x*(1-x)) - (1-x) / (1+x+2*x*(1-x)) - x / (2-x+2*x*(1-x)))
	    return value

    def true_FMado(self,x):
	    value = (1/2) * ((x / (1+x*(1-x))) * (1 - 1 / (1 + x)) + ((1 - x) / (x * (2 - x) + 1 - x)) * (1 - 1 / (2 - x)))
	    return value

    def simu(self):
        """
            Perform Monte Carlo simulation
        """

        output = []

        for k in range(self.n_iter):
            FMado_store = np.zeros(len(self.n_sample))
            obs_all = self.simulation(np.max(self.n_sample))
            for i in range(0, len(self.n_sample)):
                obs = obs_all[:self.n_sample[i]]
                FMado = self._fmado(obs)
                FMado_store[i] = FMado[0,1]
        
            output_cbind = np.c_[FMado_store, self.n_sample, np.arange(len(self.n_sample))]
            output.append(output_cbind)
        df_FMado = pd.DataFrame(np.concatenate(output))
        df_FMado.columns = ['FMado', "n", "gp"]
        df_FMado['scaled'] = (df_FMado.FMado - self.true_FMado(self.lmbd)) * np.sqrt(df_FMado.n)
        
        return(df_FMado)

    def exec_var_lmbd(self, plot = False, n_lmbd = 10):
        """
            Produce multiple monte carlo simulation for several fixed value of lmbd.

            Return a figure at end
        """

        lmbds = np.linspace(0,1,n_lmbd)
        values = self.var_mado(self.lmbd)

        fig, ax = plt.subplots(2,3, sharey = True)
        ax = ax.ravel()
        var_lmbd = []
        for lmbd in tqdm(lmbds):
            self.lmbd = lmbd
            df_FMado = self.simu()
            q_1, mean, q_9 = df_FMado['FMado_reduced'].quantile(0.05) , df_FMado['FMado_reduced'].quantile(0.5), df_FMado['FMado_reduced'].quantile(0.95)
            output = df_FMado['FMado_scaled'].var()         
            var_lmbd.append(output)
        
        ax[i].plot(x, values, '--')
        ax[i].plot(lmbds, var_lmbd, '.', markersize = 5, alpha = 0.5, color = 'salmon')
        if plot :
            plt.savefig("/home/aboulin/Documents/stage/naveau_2009/output/image_2.png")

def randomness(n):
	sample = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n)
	return sample


Monte_Carlo = simulation(n_iter = 100, simulation = randomness, n_sample = [8,16,32,64,128,256], lmbd = 0.5, random_seed = 42)
df = Monte_Carlo.simu()
print(df)
print(Monte_Carlo.var_mado(0.5))
print(df['scaled'].var())