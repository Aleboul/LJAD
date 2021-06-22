rm(list = ls())
library(MASS)
library(doRNG)
library(ggplot2)
library(dplyr)
library(VineCopula)

theme_set(theme_bw())
theme_update(axis.text.x = element_text(size = 20),
             axis.text.y = element_text(size = 20),
             axis.title.x = element_text(size = 25, margin=margin(20,0,0,0)),
             axis.title.y = element_text(size = 25, angle = 90, margin = margin(0,20,0,0)),
             legend.text = element_text(size = 20),
             legend.title = element_text(size = 20),
             title = element_text(size = 30),
             strip.text = element_text(size = 25),
             strip.background = element_rect(fill="white"),
             panel.spacing = unit(2, "lines"),
             legend.position = "bottom")

target = list()

prefix = "/home/aboulin/Documents/stage/naveau_2009/output"

theta = 15

target$generate_randomness <- function(nobservations){
	randomness = BiCopSim(nobservations, 4,theta)
	return(randomness)
}

target$robservation <- function(randomness){
	ncols = ncol(randomness)
	nrows = nrow(randomness)
	u = randomness[,1]
	v = randomness[,2]
	
	x = qexp(u)
	y = qexp(v)
	output = matrix(c(x,y), nrow = nrows, ncol = ncols)
	return(output)
}


var_FMado = function(x){
	value = ((x * (1-x)) / (1+x*(1-x)))^2 * ( 1 / (1+2*x*(1-x)) - (1-x) / (1+x+2*x*(1-x)) - x / (2-x+2*x*(1-x)))
	return(value)
}

metric_lambda = function(xvec, yvec, lambda)  mean(abs((xvec^(lambda) - yvec^(1-lambda))) - lambda * (1 - xvec^lambda) - (1-lambda) * (1 - yvec^(1-lambda))) + (1-lambda*(1-lambda)) / ((2-lambda)*(1+lambda))

simu = function(target){
	# Heuristiquement,
	# La fonction, à chaque itération (donné par foreach) va calculer un lambda-FMadogram à partir des données simulées
	# pour n = 128, 256, 512, 1024, 2048, 4096. Une fois cela fait, elle sauvegarde le résultat dans une matrice output.
	# On réitère M fois l'expérience et la fonction foreach permet d'ajouter (par lignes) les résultats dans la matrice output via 
	# le paramètre .combine = rbind. A chaque étape, la matrice output se voit ajouter 6 lignes (i.e le nombre de taille d'échantillon).
	foreach(rep = 1:M, .combine = rbind) %dorng% {
		# foreach is a function that create a loop and we, at each iteration, we increment the matrix of results (here output)
		# using rbind.
		# Allocate space for output

		FMado_store = matrix(0, length(n))
		FMado_runtimes = rep(0, length(n))
		FMado_lambda = rep(0, length(n))

		# generate all observations and sets of randomness to be used
		obs_rand = target$generate_randomness(max(n)) # we produce our bivariate vector of data
		obs_all = target$robservation(obs_rand) # some data are now hidden
		
		for(i in 1:length(n)){
			t_FMado = proc.time() # we compute the time to estimate
			# subset observations
			obs = obs_all[1:n[i],] # we pick the n[i] first rows, i.e 50 rows for the first, 100 for the second
			
			# compute the lambda FMadogram
			### build the ecdf
			ncols = ncol(obs)
			V = matrix(0, nrow = n[i], ncol = ncols) # initialisation
			for(p in 1:ncols){
				x_ = obs[,p] # we pick the data
				F_x = ecdf(x_)(x_) # we compute the ecdf using the vector of data and the vector of indicators
				V[,p] = F_x # we store the ecdf
			}
			### We compute the lambda FMadogram
			FMado = 1/2 * (metric_lambda(V[,1],V[,2], target$lambda)) # we compute now the lambda-FMadogram (the normalized one)
			t_FMado = proc.time() - t_FMado

			# Save the results
			FMado_store[i,] = FMado
			FMado_runtimes[i] = t_FMado[3]
			FMado_lambda[i] = target$lambda
		}

		output = cbind(FMado_store, FMado_runtimes,FMado_lambda, n, (1:length(n)))
		output
	}
}

A <- function(t){
	value_ = t^theta + (1-t)^theta
	return(value_^(1/theta))
}

Aprime <- function(t){
	value_1 = t^(theta-1) - (1-t)^(theta-1)
	value_2 = (t^(theta) + (1-t)^(theta))^(1/theta - 1 )
	return(value_1 * value_2)
}

Kappa <- function(t,A){
	return(A(t) - Aprime(t) * t)
}

Zeta <- function(t,A){
	return(A(t) + Aprime(t) * (1-t))
}

f <- function(t,A){
	value_ = t*(1-t) / (A(t) + t*(1-t))
	return(value_^2)
}

f1_Kappa <- function(t,A){
	value_1 = t*((1-t)^3) * Kappa(t,A)
	value_2 = (A(t) + t*(1-t)) * (A(t) + (1-t))
	return(value_1 / value_2)
}

f2_Kappa <- function(t,A){
	value_1 = t*(1-t) * Kappa(t,A)
	value_2 = A(t) - (1-t) + t*(1-t)
	value_3 = (t*(1-t)) / (A(t) + t -(1-t) + 2*t*(1-t))
	value_4 = ((1-t)^2) / (A(t) + 1-t)
	return((value_1 / value_2) * (value_3 - value_4))
}

f_Kappa <- function(t,A){
	if(t <= 0.5){
		value_1 = Kappa(t,A) * (t*(1-t))^2
		value_2 = (A(t) + t*(1-t)) * (A(t) + 2*t*(1-t))
		return(value_1 / value_2)

	}
	else{
		return(f1_Kappa(t,A) + f2_Kappa(t,A))
	}
}

f1_Zeta<- function(t,A){
	value_1 = (t^3)*(1-t) * Zeta(t,A)
	value_2 = (A(t) + t*(1-t)) * (A(t) + t)
	return(value_1 / value_2)
}

f2_Zeta <- function(t,A){
	value_1 = t*(1-t)*Zeta(t,A)
	value_2 = A(t) - t + t*(1-t)
	value_3 = (t*(1-t)) / (A(t) + (1-t) - t + 2*t*(1-t))
	value_4 = (t^2) / (A(t) + t)
	return((value_1 / value_2) * (value_3 - value_4))
}

f_Zeta <- function(t,A){
	if(t <= 0.5){
		return(f1_Zeta(t,A) + f2_Zeta(t,A))
	}
	else{
		value_1 = Zeta(t,A) * (t*(1-t))^2
		value_2 = (A(t) + t*(1-t)) * (A(t) + 2*t*(1-t))
		return(value_1 / value_2)
	}
}

V_1 <- function(t,A){
	value_ = 1 / (A(t) + 2*t*(1-t))
	return(value_)
}

V_2 <- function(t,A){
	value_1 = (Kappa(t,A)^2) * (1-t)
	value_2 = 2*A(t) - (1-t) + 2*t*(1-t)
	return(value_1 / value_2)
}

V_3 <- function(t,A){
	value_1 = (Zeta(t,A)^2) * t
	value_2 = 2*A(t) - t + 2*t*(1-t)
	return(value_1 / value_2)
}

lower <- function(t,A){
	value_1 = f(t,A)*(V_1(t,A) + V_2(t,A) + V_3(t,A))
	
	value_21 = ((1-t)^2 - A(t)) /  (2*A(t) - (1-t) + 2*t*(1-t))
	value_2 = 2 * Kappa(t,A) * f(t,A) * value_21 + 2 * f_Kappa(t,A)
	
	value_31 = (t^2 - A(t)) / (2*A(t) - t + 2*t*(1-t))
	value_3 = 2 * Zeta(t,A) * f(t,A) * value_31 + 2 * f_Zeta(t,A)
	
	value_ = value_1 - value_2 - value_3
	return(value_)
}

upper <- function(t,A){
	value_1 = f(t,A)*(V_1(t,A) + V_2(t,A) + V_3(t,A))

	value_21 = (1-t) / (2*A(t) - (1-t) + 2*t*(1-t))
	value_22 = (A(t) - t) / (A(t) + t + 2*t*(1-t))
	value_2 =  Kappa(t,A) * f(t,A) * (value_21 + value_22)
	
	value_31 = t / (2*A(t) - t + 2*t*(1-t))
	value_32 = (A(t) - (1-t)) / (A(t) + 1-t + 2*t*(1-t))
	value_3 = Zeta(t,A) * f(t,A) * (value_31 + value_32)
	
	value_41 = Zeta(t,A) * Kappa(t,A) * t*(1-t) / A(t)
	value_4 = 2 * f(t,A) * value_41
	
	value_ = value_1 - value_2 - value_3 + 2 * value_4
	return(value_)
}

M = 1000
n = c(64)
lambda = seq(0.0,1.0, length.out = 100)
lambda_FMadogram = foreach(rep = 1:length(lambda), .combine = rbind) %dorng% {
	print(rep)
	target$lambda = lambda[rep]
	prod = simu(target)
	scaled = (prod[,1] - mean(prod[,1])) * sqrt(prod[,4])
	output = cbind(prod, scaled)
}

df_FMado = data.frame(lambda_FMadogram)
names(df_FMado) = c("FMado", "runtime", "lambda", "n", "gp", "scaled")
#ggplot(mydata, aes(Var1, Var2)) + geom_point() + facet_grid(~ Variety)

var = df_FMado %>% group_by(lambda) %>% summarise(var = var(scaled))

x = seq(0,1,length.out = 100)
values_lower = rep(0,100)
values_upper = rep(0,100)

for(i in 1:100){
	t = x[i]
	values_lower[i] = lower(t,A)
}

for(i in 1:100){
	t = x[i]
	values_upper[i] = upper(t,A)
}

df_var = data.frame(x, values_lower, values_upper)

g <- ggplot(data = var, aes(x = lambda, y = var))
g <- g + geom_point(alpha = 0.75, size = 0.5, color = 'salmon')
g <- g + xlab(expression(lambda)) + ylab(expression(sigma^2)) + theme(legend.position = "none")
# g <- g + xlim(c(0,1)) + stat_function(fun = var_FMado)
g <- g + geom_line(aes(x, values_lower), df_var, linetype = "longdash", color = "lightblue") + geom_line(aes(x, values_upper), df_var, linetype = "longdash", color = "darkblue")
g
