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
	index = np.argsort(data)
	ecdf = np.zeros(len(index))
	for i in index:
		ecdf[i] = (1.0 / np.sum(missing)) * np.sum((data <= data[i]) * missing) # compute ecdf
		

	return ecdf
	
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
	col_index = np.where(missing == 1)
	X = X [:, col_index]
	dist = np.zeros([nrows, nrows])
	for i in range(0,nrows):
		for j in range(0,i):
			if i == j:
				dist[i,i] = 0
			else :
				F_x = np.squeeze(X[j,:])
				G_y = np.squeeze(X[i,:])
				d = np.linalg.norm(np.power(F_x,lmbd) - np.power(G_y,1-lmbd), ord = 1) - (lmbd) * np.sum(1-np.power(F_x,lmbd)) - (1-lmbd) * np.sum(1 - np.power(G_y, 1 - lmbd))
				dist[i,j] = d
				dist[j,i] = d
				
	return dist
	
def fmado(x, lmbd):
	"""
		This function
		
		Inputs
		------
		x : a matrix
		lmbd : constant between 0 and 1 use for the lambda F-madogram

		Outputs
		-------
		A matrix equals to 0 if i = j and equals to |F(X)^{\lambda} - G(Y)^{1-\lambda}|
	"""
	
	Nnb = x.shape[1] // 2 
	Tnb = x.shape[0]
	
	#--- Distance Matrix
	#--- F-Madogram
	V = np.zeros([Tnb, Nnb])
	for p in range(0, Nnb):
		x_vec = np.array(x[:,p]) # x
		miss = np.array(x[:, Nnb + p])
		Femp = ecdf(x_vec, miss)
		V[:,p] = Femp 
	# With Madogram
	cross_missing = np.squeeze(np.multiply(x[:,Nnb], x[:, Nnb + 1]))
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
				- probs_missing = array that indicate the probabilities of missing
				- n_sample = array of multiple lengths of sample

		Outputs
		-------
		Array containing niter * length(n_sample) estimators of the FMadogram

	"""
	output = []

	for k in range(target['niter']):
		probs = [] ; length = []
		FMado_store = np.zeros(len(target['n_sample']))
		obs_all = target['simulation'](mean, cov, np.max(target['n_sample']))
		I = np.transpose([ np.random.binomial(1, 1-p, np.max(target['n_sample'])) for p in target['probs_missing'] ])
		obs_all = np.concatenate([obs_all, I], axis = 1)
		for i in range(0, len(target['n_sample'])):
			obs = obs_all[:target['n_sample'][i]]
			FMado, l = fmado(obs, target['lambda'])
			FMado_store[i] = FMado[0,1] 
			probs.append(target['probs_missing']) ; length.append(l)

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
				- probs_missing = array that indicate the multiple probabilities of missing
				- n_sample = length of sample

		Outputs
		-------
		Array containing niter * length(probs_missing) estimators of the FMadogram
	"""

	output = []
	for k in tqdm(range(target['niter'])):
		probs = []
		FMado_store = np.zeros(len(target['probs_missing']))
		obs_all = target['simulation'](mean, cov, np.max(target['n_sample']))
		for i in range(0, len(target['probs_missing'])):
			I = np.transpose([ np.random.binomial(1,1-p, np.max(target['n_sample'])) for p in target['probs_missing'][i] ])
			obs = np.concatenate([obs_all, I], axis = 1)
			FMado = fmado(obs)
			FMado_store[i] = FMado[0,1]
			probs.append(target['probs_missing'][i])
		output_cbind = np.c_[FMado_store, np.repeat(target['n_sample'], len(target['probs_missing'])),np.arange(len(target['probs_missing'])), probs]
		output.append(output_cbind)

	return output

def var_mado_missing(x, p_xy, p_x, p_y):
	value = ((x ** 2 * (1-x)**2) / (1+x*(1-x))**2) * ( (p_xy**-1) / (1+2*x*(1-x)) - (p_x**-1)* (1-x) / (1+x+2*x*(1-x)) - (p_y**-1)*x / (2-x+2*x*(1-x)))
	return value
target = {}

target['niter'] = 100
target['simulation'] = np.random.multivariate_normal
target['probs_missing'] = [0.1,0.1]
n_sample = [100,250,500,1000,10000]

mean = [0,0]
cov = [[1,0],[0,1]]

lmbds = np.linspace(0,1,50)
x = np.linspace(0, 1, 50)
values = var_mado_missing(x,(1- target['probs_missing'][0])*(1- target['probs_missing'][1]), 1-target['probs_missing'][0], 1-target['probs_missing'][1])  

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

plt.savefig("/home/aboulin/Documents/stage/naveau_2009/output_2.png")