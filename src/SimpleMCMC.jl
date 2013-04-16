module SimpleMCMC


include("parsing.jl") #  include model processing functions		
include("diff.jl") #  include derivatives definitions
include("distribs.jl") #  include distributions definitions
include("samplers.jl") #  include sampling functions


import Base.show

export simpleRWM, simpleHMC, simpleNUTS
export buildFunction, buildFunctionWithGradient

# naming conventions
const ACC_SYM = :__acc
const PARAM_SYM = :__beta
const LLFUNC_NAME = "loglik"
const TEMP_NAME = "tmp"
const DERIV_PREFIX = "d"


end # module end

