
include("../src/SimpleMCMC.jl")


logpdfNormal(0,1,1)
SimpleMCMC.logpdfNormal(0,1,1)

logpdfNormal(1.5,2,10)
SimpleMCMC.logpdfNormal(1.5,2,10)

logpdfNormal(1.5,-2,10)
SimpleMCMC.logpdfNormal(1.5,-2,10)


a = 0. ; b = 1.; c = 1. ; tic() ; for i in 1:10000000; logpdfNormal(a,b,c) ; end ; toq()   # 1.87 s
a=zeros(10); b=ones(10); c=ones(10); tic() ; for i in 1:1000000; logpdfNormal(a,b,c) ; end ; toq()   #  4.75 s , * 2.54
a=zeros(100); b=ones(100); c=ones(100); tic() ; for i in 1:100000; logpdfNormal(a,b,c) ; end ; toq()   #  4.66 s , * 2.49
a=zeros(1000); b=ones(1000); c=ones(1000); tic() ; for i in 1:10000; logpdfNormal(a,b,c) ; end ; toq()   #  4.66 s , * 2.49

a = [0.] ; b = 1.; c = 1. ; tic() ; for i in 1:10000000; logpdfNormal(a,b,c) ; end ; toq()   # 6.85 s, * 3.6, 2.04s
a = [0.] ; b = [1.]; c = 1. ; tic() ; for i in 1:10000000; logpdfNormal(a,b,c) ; end ; toq()   # 6.60 s, 2.23
a = [0.] ; b = [1.]; c = [1.] ; tic() ; for i in 1:10000000; logpdfNormal(a,b,c) ; end ; toq()   # 5.82 s

a = zeros(1000) ; b = 1.; c = 1. ; tic() ; for i in 1:10000; logpdfNormal(a,b,c) ; end ; toq()   # 5.82 s



res = SimpleMCMC.simpleRWM(:(x::real;x~Normal(0,1)), 10000, 0)
open("c:/temp/sampnor.txt", "w") do io
    println(io, join(keys(res.params), '\t'))
    writedlm(io, hcat([s[2] for s in res.params]...))
end

res = SimpleMCMC.simpleRWM(:(x::real;x~Uniform(0,2)), 10000, 0)
open("c:/temp/sampnor.txt", "w") do io
    println(io, join(keys(res.params), '\t'))
    writedlm(io, hcat([s[2] for s in res.params]...))
end

res = SimpleMCMC.simpleRWM(:(x::real;x~Gamma(2,1)), 10000, 0)
open("c:/temp/sampnor.txt", "w") do io
    println(io, join(keys(res.params), '\t'))
    writedlm(io, hcat([s[2] for s in res.params]...))
end

###############################################☺

include("../src/SimpleMCMC.jl")

b = 12
SimpleMCMC.generateModelFunction(:(a::real; a+b), 1.0, false, true)
SimpleMCMC.generateModelFunction(:(a::real; a+b), 1.0, true, true)


SimpleMCMC.simpleAGD(:(a::real; -(a+b)^2))
res = SimpleMCMC.simpleAGD(:(x::real; -log(x)*log(x)-x), 1.)
res = SimpleMCMC.simpleAGD(:(x::real; -log(x)*log(x)-x), 0.5)
res = SimpleMCMC.simpleAGD(:(x::real; -log(x)*log(x)-x), 2.)
res = SimpleMCMC.simpleAGD(:(x::real; -log(x)*log(x)-x), 5.)
dump(res)
log(-1.)


m = SimpleMCMC.parseModel(:(a::real; a+b))
SimpleMCMC.unfold!(m)
m.exprs
SimpleMCMC.uniqueVars!(m)
SimpleMCMC.categorizeVars!(m)

m

SimpleMCMC.preCalculate(m)



        backwardSweep!(m)


################################################################

template = quote
    function logpdfNormal(x1::AbstractArray, x2::Real)
        local res = 0.0
        for i in 1:length(x1) ; res += logpdfNormal(x1[i], x2) ; end
        res
    end
end

template.args[2].args[1].head
dump(template.args[2].args[2].args[4])

mf = expr(:function, 
          expr(:call, :logpdfNormal, 
                expr(:(::), :x1, :AbstractArray), 
                expr(:(::), :x2, :Real)),
          expr(:block, 
                :(local res = 0.),
                expr(:for, 
                     expr(:(=), :i, expr(:(:), 1, :(length(x1)))),
                     expr(:block, 
                          expr(:(+=), :res, 
                          expr(:call, :logpdfNormal, 
                                      :(x1[i]),
                                      :(x2)))
                         )
                    ), 
                 :(res) 
                ) 
           )

mf
template


pars = [(:x1, :AbstractArray), (:x2, :Real)]
rv = pars[find([p[2] for p in pars] .== :AbstractArray)][1][1]

pars[find([p[2] for p in pars] .== :AbstractArray)][1,1]
size(pars[find([p[2] for p in pars] .== :AbstractArray)])

mf = expr(:function, 
          expr(:call, :logpdfNormal, 
                [expr(:(::), p...) for p in pars]...),
          expr(:block, 
                :(local res = 0.),
                expr(:for, 
                     expr(:(=), :i, expr(:(:), 1, :(length($rv)))),
                     expr(:block, 
                          expr(:(+=), :res, 
                          expr(:call, :logpdfNormal, 
                                      [p[2]==:Real ? p[1] : expr(:ref, p[1], :i) for p in pars]...))
                         )
                    ), 
                 :(res) 
                ) 
           )
