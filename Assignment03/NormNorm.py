import numpy as np
import scipy
from scipy import stats, integrate
import matplotlib.pyplot as plt
from scipy import *
import math


#generate a distribution
sample_size = 100
pop_mean = 60
pop_std = 50
samp_dist= stats.norm(loc=pop_mean, scale = pop_std)


#draw a sample from the distribution
np.random.seed(seed=233423)
sample= samp_dist.rvs(size = sample_size)
sample_w = sample.std()/ sqrt(sample_size)

#prior mean and standard deviation
prior_mean = 80
prior_std = 50
prior_w = prior_std/sqrt(sample_size) 


#Calculating Posterior
B = (sample_w**2)/(sample_w**2+prior_w**2)
mean_post = sample.mean() + B *(prior_mean-sample.mean())
w_post = sample_w * sqrt(1-B)


#mu values to evaluate pdf
mu_array = linspace(10.,150.,300)

posterior = stats.norm.pdf(mu_array,loc = mean_post, scale = w_post)


#evaluate normal prior and liklihood on mu_array
prior = stats.norm.pdf(mu_array, loc = prior_mean, scale = prior_w)
liklihood = stats.norm.pdf(mu_array, loc = sample.mean(), scale = sample_w)


#define trapizoid function
def trapIntegrate(funct, delta):
	'''
	compute the area of a function
	'''
	sumInt = funct[0]
	for i in range (1,funct.size-2):
		sumInt = sumInt + 2 * funct[i]
	sumInt = sumInt + funct[funct.size-1]
	return sumInt*delta/2

mlike = trapIntegrate(prior*liklihood,(140/300))
# mlike = integrate.trapz(prior *liklihood, mu_array)

post_pdf = prior*liklihood/mlike


plt.plot(mu_array, posterior, lw=3, label = "Posterior 4.1")
plt.plot(mu_array,post_pdf, 'r:', label = 'Posterior 4.2')
plt.xlabel(r'$\mu$')
plt.ylabel('p($\mu$|...)')
plt.legend(loc = 'upper right')
plt.show()



import numpy.testing as npt

def test_trap():
	'''
	test the implementation of trapezoidal rule
	'''
	#test the trapezoidal rule 
	npt.assert_allclose(mlike, integrate.trapz(prior*liklihood,mu_array), rtol = 1e-2)
	
def test_post():
	'''
	test the implementation of trapezoidal rule
	'''
	#test the trapezoidal rule 
	npt.assert_allclose(posterior,post_pdf, rtol = 1e-2)

test_trap()
test_post()



