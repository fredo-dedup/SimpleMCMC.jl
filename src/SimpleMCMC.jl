module SimpleMCMC


import Base.show

import Base.sum
sum(x::Real) = x  # meant to avoid the annoying behaviour of sum(Inf) 

export simpleRWM, simpleHMC, simpleNUTS
export simpleAGD
export buildFunction, buildFunctionWithGradient

# naming conventions
const ACC_SYM = :__acc       # name of accumulator variable
const PARAM_SYM = :__beta    # name of parameter vector
const TEMP_NAME = "tmp"      # prefix of temporary variables in log-likelihood function
const DERIV_PREFIX = "d"     # prefix of gradient variables in log-likelihood function

include("parsing.jl")      #  include model processing functions		
include("diff.jl")         #  include derivatives definitions
include("distribs.jl")     #  include distributions definitions
include("samplers.jl")     #  include sampling functions
include("solvers.jl")      #  include solving functions

end # module end

