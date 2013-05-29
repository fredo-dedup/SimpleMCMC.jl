
include("../src/SimpleMCMC.jl")

include("../src/distribs.jl")
logpdfNormal

a = 0. ; b = 1.; c = 1. ; tic() ; for i in 1:10000000; logpdfNormal(a,b,c) ; end ; toq()   # 1.87 s
a=zeros(10); b=ones(10); c=ones(10); tic() ; for i in 1:1000000; logpdfNormal(a,b,c) ; end ; toq()   #  4.75 s , * 2.54
a=zeros(100); b=ones(100); c=ones(100); tic() ; for i in 1:100000; logpdfNormal(a,b,c) ; end ; toq()   #  4.66 s , * 2.49
a=zeros(1000); b=ones(1000); c=ones(1000); tic() ; for i in 1:10000; logpdfNormal(a,b,c) ; end ; toq()   #  4.66 s , * 2.49

a = [0.] ; b = 1.; c = 1. ; tic() ; for i in 1:10000000; logpdfNormal(a,b,c) ; end ; toq()   # 6.85 s, * 3.6, 2.04s
a = [0.] ; b = [1.]; c = 1. ; tic() ; for i in 1:10000000; logpdfNormal(a,b,c) ; end ; toq()   # 6.60 s, 2.23
a = [0.] ; b = [1.]; c = [1.] ; tic() ; for i in 1:10000000; logpdfNormal(a,b,c) ; end ; toq()   # 5.82 s

a = zeros(1000) ; b = 1.; c = 1. ; tic() ; for i in 1:10000; logpdfNormal(a,b,c) ; end ; toq()   # 5.82 s


res = SimpleMCMC.simpleRWM(:(x::real;x~Gamma(2,1)), 10000, 0)
open("c:/temp/sampnor.txt", "w") do io
    println(io, join(keys(res.params), '\t'))
    writedlm(io, hcat([s[2] for s in res.params]...))
end

###############################################☺

include("../src/SimpleMCMC.jl")

SimpleMCMC.simpleAGD(:(a::real; -(a+b)^2))
res = SimpleMCMC.simpleAGD(:(x::real; -log(x)*log(x)-x), 1.)
res = SimpleMCMC.simpleAGD(:(x::real; -log(x)*log(x)-x), 0.5)
res = SimpleMCMC.simpleAGD(:(x::real; -log(x)*log(x)-x), 2.)
res = SimpleMCMC.simpleAGD(:(x::real; -log(x)*log(x)-x), 5.)
dump(res)


################################################################
