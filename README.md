simple-mcmc
===========

Basic mcmc samplers written in Julia

Implements :
1. a DSL for specifying models
2. a set of sampling functions

## The model DSL
Basically the full Julia language can be used with the following limitations :
- the `::` operator is redefined to declare model parameters being sampled, this makes the usual meaning of the operator inaccessible within the model definition
- the '~' operator is redefined to associate model variables with a distribution, with the same consequences
- as the model parsing creates a new function in the 'Main' namespace (__loglik), and introduces new variables within the model (__acc and __beta) there are potential variables collisions : don't use those names
- sampling functions that use a gradient (HMC) are limited to models using functions/operators for which partial derivatives are specified. The list is easily extendable (see function 'derive'). Note too that control flow operators (for, if, while) are not yet implemented and will necessitate a bit more work..

An example model spec should be enough to illustrate the DSL : 

`
model = quote
	b::scalar
	k::vector(5)
	
	a = b+6
	x = sin(dot(k, z))

	x ~ Weibull(a, 2.0)
end
`

- `::` indicates variables to be sampled : `b::scalar` declares a scalar model parameter b, `k::vector(5)` declares a vector model parameter of size 5. The size can be also be an expression as long as it is evaluates to a stricly positive integer (an error will be thrown otherwise).
- Statements with the operator ~ (`x ~ Weibull(a, 2.0)`) declare how to build the model likelihood, here this says that x should have a Weibull distributions (any continuous distribution of the "Distribution.jl" module can be used) of shape `a` and scale `2`
- other statements are evaluated normally and can either use and define model-local variables (`a`, `x`) or use variables defined in the calling environment (`z`)

## The sampling functions
Currently, we have `simpleRWM` running a random walk metropolis and `simpleHMC` running an Hamiltonian Monte-Carlo with a reverse mode gradient calculation (for the curious you can run SimpleMCMC.buildFunctionWithGradient(model) to get the generated code).

Calling syntax
- simpleRWM(model::Expr, steps::Integer, burnin::Integer, init::Any)
- simpleRWM(model::Expr, steps::Integer, burnin::Integer) : with inital values set to 1.0
- simpleRWM(model::Expr, steps::Integer) : with burnin equal to half of steps

- simpleHMC(model::Expr, steps::Integer, burnin::Integer, init::Any, length::Integer, stepsize::Float64)
- simpleHMC(model::Expr, steps::Integer, burnin::Integer, length::Integer, stepsize::Float64) : with inital values set to 1.0
- simpleHMC(model::Expr, steps::Integer, length::Integer, stepsize::Float64) : with burnin equal to half of steps

## Examples


## Improvements
- Add adaptative algorithms for simpleHMC for length and stepsize parameters (using NUTS ?)
- Convert the set of jags/bugs examples, run them and compare timings
- See if optimizing the generated gradient code is spossible
- Specify partial derivatives of other continuous distributions (currently only Normal, Uniform and Weibull)
- Add truncation
- Add validity domains for model parameters
- Add discrete distributions ?

