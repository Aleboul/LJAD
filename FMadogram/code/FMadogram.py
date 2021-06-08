import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from matplotlib import cm
from tqdm import tqdm

def ecdf(data, missing):
	""" Compute ECDF  with missing data 
		
		Inputs
		------
		data : array of data
		missing : index of missing data
		
		Outputs
		-------
		empirical cumulative distribution function
	"""
	index = np.argsort(data) # gives index of the sorted vector (for example [1,4,3,2] is [0,2,3,1])
	ecdf = np.zeros(len(index)) # length of the vector of index
	for i in index:
		ecdf[i] = (1.0 / np.sum(missing)) * np.sum((data <= data[i]) * missing) # compute ecdf(data) (same duty as ecdf R function )
		

	return ecdf # ecdf(data)
	
def dist(X, lmbd, missing):
	"""
		Compute matrix of F-Madogram (a une constante pres) using the empirical cumulative distribution function
		
		Inputs
		------
		X : a matrix composed of ecdf
		lmbd : a parameter between 0 and 1
		missing : vector indicate if we have the complete value

		Outputs
		-------
		A matrix with quantity equals to 0 if i=j (diagonal) and equals to sum_t=1^T |F(X_t)^{\lambda} - G(Y_t)^{1-\lambda}| if i \neq j
	"""
	ncols = X.shape[1]
	nrows = X.shape[0]
	dist = np.zeros([nrows, nrows]) # intialization
	for i in range(0,nrows):
		for j in range(0,i):
			if i == j:
				dist[i,i] = 0 
			else :
				F_x = np.squeeze(X[j,:]) * missing # squeeze return an array, we had a list of array before
				G_y = np.squeeze(X[i,:]) * missing
				d = np.linalg.norm(np.power(F_x,lmbd) - np.power(G_y,1-lmbd), ord = 1) - (lmbd) * np.sum(missing-np.power(F_x,lmbd)) - (1-lmbd) * np.sum(missing - np.power(G_y, 1 - lmbd)) # formula of the normalized lambda madogram estimator, see Naveau 2009
				dist[i,j] = d
				dist[j,i] = d
				
	return dist
	
def fmado(x, lmbd):
	"""
		This function computes the lambda FMadogram
		
		Inputs
		------
		x : a matrix
		lmbd : constant between 0 and 1 use for the lambda F-madogram

		Outputs
		-------
		A matrix equals to 0 if i = j and equals to |F(X)^{\lambda} - G(Y)^{1-\lambda}|
	"""
	
	Nnb = x.shape[1] // 2 # 2 means the dimension of our vector, here, we work with bivariate sample (X_t, Y_t), we divide by two because the observations are (X_t, I_t, Y_t, J_t)
	Tnb = x.shape[0]	  # Number of observations
	
	#--- Distance Matrix
	#--- F-Madogram
	V = np.zeros([Tnb, Nnb]) # Initialization
	for p in range(0, Nnb):
		x_vec = np.array(x[:,p]) # vector of observations
		miss = np.array(x[:, Nnb + p]) # vector which indicate missing data
		Femp = ecdf(x_vec, miss) # we compute the ecdf using the missing data
		V[:,p] = Femp # we stock it
	# With Madogram
	cross_missing = np.squeeze(np.multiply(x[:,Nnb], x[:, Nnb + 1])) # Multiply the vector I and J in order to have the rows in which data are completely observed
	Fmado = dist(np.transpose(V),lmbd = lmbd, missing = cross_missing) / (2 * np.sum(cross_missing)) + (1/2) * ((1 - lmbd*(1-lmbd))/ ((2-lmbd)*(1+lmbd)))
	
	return Fmado , np.sum(cross_missing)

def simu(target):
	"""
		Perform multiple simulation of the estimation of FMadogram with an increasing length of sample

		Inputs
		------

		target : a list which contain the following parameters
				- niter = number of replication
				- simulation = law that generate the data
				- presence = array that indicate the probabilities of missing
				- n_sample = array of multiple lengths of sample

		Outputs
		-------
		Array containing niter * length(n_sample) estimators of the FMadogram

	"""
	output = [] # initialisation

	for k in range(target['niter']):
		probs = [] ; length = [] # probs collect probability of presence, length collect the real length of the sample (in case of no missing data T / n = length)
		FMado_store = np.zeros(len(target['n_sample']))
		obs_all = target['simulation'](np.max(target['n_sample']))
		I = np.transpose([ np.random.binomial(1, p, np.max(target['n_sample'])) for p in target['presence'] ])
		obs_all = np.concatenate([obs_all, I], axis = 1)
		for i in range(0, len(target['n_sample'])):
			obs = obs_all[:target['n_sample'][i]]
			FMado, l = fmado(obs, target['lambda'])
			FMado_store[i] = FMado[0,1] 
			probs.append(target['presence']) ; length.append(l)

		output_cbind = np.c_[FMado_store, target['n_sample'], np.arange(len(target['n_sample'])),length,probs]
		output.append(output_cbind)

	return output

def simu_proba(target):
	"""
		Perform multiple simulation of the FMadogram estimator with an increase of the probability of missing

		Inputs
		------

		target : a list which contain the following parameters
				- niter = number of replication
				- simulation = law that generate the data
				- presence = array that indicate the multiple probabilities of missing
				- n_sample = length of sample

		Outputs
		-------
		Array containing niter * length(presence) estimators of the FMadogram
	"""

	output = []
	for k in tqdm(range(target['niter'])):
		probs = []
		FMado_store = np.zeros(len(target['presence']))
		obs_all = target['simulation'](np.max(target['n_sample']))
		for i in range(0, len(target['presence'])):
			I = np.transpose([ np.random.binomial(1,p, np.max(target['n_sample'])) for p in target['presence'][i] ])
			obs = np.concatenate([obs_all, I], axis = 1)
			FMado = fmado(obs)
			FMado_store[i] = FMado[0,1]
			probs.append(target['presence'][i])
		output_cbind = np.c_[FMado_store, np.repeat(target['n_sample'], len(target['presence'])),np.arange(len(target['presence'])), probs]
		output.append(output_cbind)

	return output

def var_mado_missing(x, p_xy, p_x, p_y):
	value = ((x ** 2 * (1-x)**2) / (1+x*(1-x))**2) * ( (p_xy**-1) / (1+2*x*(1-x)) - (p_x**-1)* (1-x) / (1+x+2*x*(1-x)) - (p_y**-1)*x / (2-x+2*x*(1-x)))
	return value

def randomness(n):
		sample = np.random.multivariate_normal([0,0], [[1,0],[0,1]], n)
		return sample

def exec_varlambda(plot = False):
	target = {}

	target['niter'] = 1000
	target['simulation'] = randomness
	target['presence'] = [1.0,1.0]
	n_sample = [2,4,8,16,32,64]

	mean = [0,0]
	cov = [[1,0],[0,1]]

	lmbds = np.linspace(0,1,1000)
	x = np.linspace(0, 1, 100)
	values = var_mado_missing(x,(target['presence'][0])*(target['presence'][1]), target['presence'][0], target['presence'][1])  

	fig, ax = plt.subplots(2,3, sharey = True)
	ax = ax.ravel()
	for i,n in enumerate(n_sample):
	  target['n_sample'] = [n]
	  var_lambda = []
	  for lmbd in tqdm(lmbds):
	    target['lambda'] = lmbd
	    output = simu(target)
	    df_FMado = pd.DataFrame(np.concatenate(output))
	    df_FMado.columns = ['FMado', 'n', 'length','gp', 'prob_X', 'prob_Y']
	    df_FMado['FMado_scaled'] = (df_FMado.FMado - df_FMado.groupby('n')['FMado'].transform('mean')) * np.sqrt(df_FMado.n)
	    output = df_FMado['FMado_scaled'].var()
	    var_lambda.append(output)
	  ax[i].plot(x, values, '--')
	  ax[i].plot(lmbds, var_lambda, '.', markersize = 5, alpha = 0.5, color = 'salmon')
	
	if plot :

		plt.savefig("/home/aboulin/Documents/stage/naveau_2009/output/image_2.png")

def exec_varn(plot = False):
	target = {}
	mean = [0,0]
	cov = [[1,0],[0,1]]

	target['niter'] = 100
	target['simulation'] = randomness
	target['presence'] = [0.9,0.9]
	target['lambda'] = 0.5
	n_sample = [x for x in range(100,10000,100)]

	values = var_mado_missing(target['lambda'], target['presence'][0] * target['presence'][1], target['presence'][0], target['presence'][1])
	var_n = []
	for n in tqdm(n_sample):
		target['n_sample'] = [n]
		output = simu(target)
		df_FMado = pd.DataFrame(np.concatenate(output))
		df_FMado.columns = ['FMado', 'n', 'length', 'gp', 'prob_X', 'prob_Y']
		df_FMado['FMado_scaled'] = (df_FMado.FMado - df_FMado.groupby('n')['FMado'].transform('mean')) * np.sqrt(df_FMado.n)
		output = df_FMado['FMado_scaled'].var()
		var_n.append(output)

	fig, ax = plt.subplots()
	ax.plot(n_sample, np.repeat(values,len(n_sample)), '--')
	ax.plot(n_sample, var_n, ".", markersize = 5, alpha = 0.5, color = "salmon")

	if plot :
		plt.savefig("/home/aboulin/Documents/stage/naveau_2009/output/image_1.png")

exec_varlambda(plot = True)