rm(list = ls())
library(MASS)
library(doRNG)
library(ggplot2)
library(dplyr)

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
target$generate_randomness <- function(nobservations){
	mean = c(0,0)
	cov  = matrix(c(1,0,0,1), ncol = 2, nrow = 2)
	return(mvrnorm(nobservations, mu = mean, Sigma = cov))
}

target$robservation <- function(presence, randomness){
	ncols = ncol(randomness)
	nrows = nrow(randomness)
	output = randomness
	for(p in 1:ncols){
		indicators = rbinom(nrows, 1, presence[p])
		output = cbind(output, indicators)
	}
	return(output)
}

compute_ecdf = function(xvec, missing){
	index = order(xvec)
	ecdf = rep(0, length(xvec))
	for(i in index){
		ecdf[i] = (1/sum(missing)) * sum((xvec <= xvec[i]) * missing)
	}
	
	return(ecdf)
}
metric_lambda = function(xvec, yvec, lambda, cross) (1 / sum(cross)) * sum(abs((xvec^(lambda) - yvec^(1-lambda))*cross) - lambda * (1 - xvec^lambda) * cross - (1-lambda) * (1 - yvec^(1-lambda)) * cross) + (1-lambda*(1-lambda)) / ((2-lambda)*(1+lambda))

true_FMado = function(lambda){
	value = (1/2) * ((lambda) / (1+lambda*(1-lambda)) * (1 - 1 / (1 + lambda)) + (1 - lambda) / (lambda * (2 - lambda) + 1 - lambda) * (1 - 1 / (2 - lambda)))
	return(value)
}

var_FMado = function(x, p_xy, p_x, p_y){
	value = ((x * (1-x)) / (1+x*(1-x)))^2 * ( (p_xy^-1) / (1+2*x*(1-x)) - (p_x^-1)* (1-x) / (1+x+2*x*(1-x)) - (p_y^-1)*x / (2-x+2*x*(1-x)))
	return(value)
}

simu = function(target){
	foreach(rep = 1:M, .combine = rbind) %dorng% {
		# Allocate space for output

		FMado_store = matrix(0, length(n))
		FMado_runtimes = rep(0, length(n))
		FMado_lambda = rep(0, length(n))

		# generate all observations and sets of randomness to be used
		obs_rand = target$generate_randomness(max(n))
		obs_all = target$robservation(target$presence, obs_rand)

		for(i in 1:length(n)){
			t_FMado = proc.time()
			# subset observations
			obs = obs_all[1:n[i],]

			# compute the lambda FMadogram
			### build the ecdf
			ncols = ncol(obs) %/% 2 # we divide by 2 due to indicators functions
			V = matrix(0, nrow = n[i], ncol = ncols)
			cross_ = rep(1, n[i])
			for(p in 1:ncols){
				x_ = obs[,p]
				miss_ = obs[, ncols + p]
				F_x = compute_ecdf(x_, miss_)
				V[,p] = F_x
				cross_ = cross_ * miss_
			}
			### We compute the lambda FMadogram
			FMado = 1/2 * (metric_lambda(V[,1],V[,2], target$lambda, cross_))
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
target$presence = c(0.7,0.7)
M = 100
n = c(32,128, 256, 512, 1024)
lambda = seq(0.0, 1.0, length.out = 10)

filename <- paste0(prefix,"fmado",target$lambda,"_M",M,".RData")
lambda_FMadogram = foreach(rep = 1:length(lambda), .combine = rbind) %dorng% {
	print(rep)
	target$lambda = lambda[rep]
	prod = simu(target)
	scaled = (prod[,1] - true_FMado(target$lambda)) * sqrt(prod[,4])
	output = cbind(prod, scaled)
}
df_FMado = data.frame(lambda_FMadogram)
names(df_FMado) = c("FMado", "runtime", "lambda", "n", "gp", "scaled")
save(df_FMado,t,file = filename)
print(df_FMado)
#ggplot(mydata, aes(Var1, Var2)) + geom_point() + facet_grid(~ Variety)

var = df_FMado %>% group_by(lambda, n) %>% summarise(var = var(scaled))

print(var)
print(df)
g <- ggplot(data = var, aes(x = lambda, y = var, colour = n, group = n))
g <- g + geom_point(alpha = 0.5, size = 0.25)
g <- g + scale_colour_gradient2(midpoint = n[length(n) - 1])
g <- g + xlab(expression(lambda)) + ylab(expression(sigma)) + theme(legend.position = "none")
g <- g + xlim(c(0,1)) + stat_function(fun = var_FMado, args = list(p_xy = 0.7*0.7, p_x = 0.7, p_y = 0.7))
g

ggsave(filename = paste0(prefix, "var_lambda_miss.png"), plot = g, width = 7, height = 5, dpi = 300)