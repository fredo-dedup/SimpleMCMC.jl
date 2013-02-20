######### hierarchical logistic regression  ###########
# using example of STAN 1.0.2 manual, chapter 10.7 p. 49

data {
int<lower=1> D;
int<lower=0> N;
int<lower=1> L;
int<lower=0,upper=1> y[N];
int<lower=1,upper=L> ll[N];
row_vector[D] x[N];
}
parameters {
real mu[D];
real<lower=0> sigma[D];
vector[D] beta[L];
}

model {
for (d in 1:D) {
	mu[d] ~ normal(0,100);
	sigma[d] ~ uniform(0,1000);

	for (l in 1:L) 	beta[l,d] ~ normal(mu[d],sigma[d]);
}

	for (n in 1:N) y[n] ~ bernoulli(inv_logit(x[n] * beta[ll[n]]));
}


N = 50 # number of observations
D = 4  # number of groups
L = 5  # number of predictors

ll = int(rand(N)*D) + 1  # mapping obs -> group
x = randn((N, L))  # predictors

extender = ones((1,L))
model = quote
	mu::real(D)
	sigma::real(D)
	beta::real(L*D)

	mu ~ Normal(0, 100)
	sigma ~ Uniform(0, 1000)

	beta ~ Normal(mu * extender, sigma * extender)

	prob = 1. / ( 1. + exp(- x * reshape(beta[ll], L, N) ) )
	y ~ Normal(prob, 1)   #Bernoulli(prob)
end

SimpleMCMC.simpleRWM(model, 10)
SimpleMCMC.simpleNUTS(model, 10)


