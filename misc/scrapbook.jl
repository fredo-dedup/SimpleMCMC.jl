load("pkg")
Pkg.init()
Pkg.add("Distributions")
Pkg.update("Distributions")
Pkg.add("DataFrames")

require("../../.julia/Distributions.jl/src/Distributions.jl")
using Distributions
################################

include("../src/SimpleMCMC.jl")

function recap(res)
    print("accept : $(round(100*mean(res[:,2]),1))%, ")
    print("mean : $(round(mean(res[:,3]),3)), ")
    println("std : $(round(std(res[:,3]),3))")
end

model = :(x::real ; x ~ Weibull(1, 1))  # mean 1.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [1.]))  # 3.400 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [1.], 1, 0.001)) # 9.500 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [1.]))  # 400-500 ess/s, mean is off by -0.05

model = :(x::real ; x ~ Weibull(3, 1)) # mean 0.89, std 0.325
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [1.]))  # 5.900 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.6], 1, 0.01)) # 30.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [1.]))  # 15.000 ess/s, correct

model = :(x::real ; x ~ Uniform(0, 2)) # mean 1.0, std 0.577
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [1.]))  # 5.000 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [1.], 2, 0.7)) # 14.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 10000, 1000, [1.]))  # 130 ess/s, very slow due to gradient == 0 ?

model = :(x::real ; x ~ Normal(0, 1)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.]))  # 6.000 ess/s  7500
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.], 2, 0.8)) # 45.000 ess/s, 49-50.000
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.2]))  # 20.000 ess/s, correct, 22-24.0000

model = :(x::real ; x ~ Normal(3, 12)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.]))  # 6.200 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.], 2, 0.8)) # 45.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.]))  # 17.000 ess/s, correct

model = :(x::real(10) ; x ~ Normal(3, 12)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, 0.))  # 300 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, 0., 2, 0.8)) # 45.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, 0.))  # 4.800 ess/s, correct, (4.600 without direct call to libRmath)


z = Float64[ randn(10) .< 0.5 ]
model = :(x::real ; z ~ Bernoulli(x)) # mean 0.5, std ...
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.5]))  # 6.200 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.5], 2, 0.04)) # 140 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.5]))  # 17.000 ess/s, correct




model = :(x::real ; x ~ Weibull(1, 1)) 
recap(SimpleMCMC.simpleNUTS(model, 10000, 0, [0.2]))  
model = :(x::real ; x ~ Normal(0, 1)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleNUTS(model, 10000, 0, [0.2]))  

res = SimpleMCMC.simpleNUTS(model, 10000, 0, [0.2])
mean(res[:,3])

0ÓK], -0.9389385332046728, -0.2 
 4.0 
 0.7, 1000, 0.05, 0.75, 10 
 0.0, 3.6888794541139363, 0.0

p!ç], -0.9389385332046728, -0.2 
 4.0 
 0.7, 1000, 0.05, 0.75, 10 
 0.0, 3.6888794541139363, 0.0

__loglik(ones(10)*3)[1]
__loglik([1.])[1]
__loglik([-1.])[1]
__loglik([0.636822])[1]
__loglik([0.636822])[2]
__loglik([-0.636822])[1]
__loglik([-0.636822])[2]

res = SimpleMCMC.simpleNUTS(model, 100000, 1000, [1.])
dlmwrite("c:/temp/dump.txt", res[:,3])

########################################################################
myf, np = SimpleMCMC.buildFunctionWithGradient(model)
eval(myf)

res = SimpleMCMC.simpleHMC(model,100, 5, 1.5, 10, 0.1)

l0, grad = __loglik(ones(2))
[ [ (beta=ones(2) ; beta[i] += 0.01 ; ((__loglik(beta)[1]-l0)*100)::Float64) for i in 1:2 ] grad]


################################################

Y = [1,2,3]
model = :(x::real(3); y=Y; y[2] = x[1] ; y ~ TestDiff())

## ERROR : marche pas, notamment parce que la subst des noms de variable ne fonctionne pas

################################


model = quote
    x::real
    x ~ Normal(0,1)
end

res = SimpleMCMC.simpleRWM(model, 10000)
res2 = {:llik=>[1,2,3,3,4], :accept=>[0,1,0,1,0,1,1,1,0,0,0],
:x=>[4,4,4,5,5,5,6,6,3,22,3]}
mean(res2[:accept])
dump(res2)

res = SimpleMCMC.simpleHMC(model, 10000, 5000, 3, 0.5)
[ (mean(res[:,i]), std(res[:,i])) for i in 3:size(res,2)]
=======
dump(:(b = x==0 ? x : y))
dump(:(x==0 ))
dump(:(x!=0 ))
dump(:(x!=0 ))


################################


dat = dlmread("c:/temp/ts.txt")
size(dat)

tx = dat[:,2]
dt = dat[2:end,1] - dat[1:end-1,1]

model = quote
    mu::real
    tau::real
    sigma::real

    tau ~ Weibull(2,1)
    sigma ~ Weibull(2, 0.01)
    mu ~ Uniform(0,1)

    f2 = exp(-dt / tau / 1000)
    resid = tx[2:end] - tx[1:end-1] .* f2 - mu * (1.-f2)
    resid ~ Normal(0, sigma^2)
end

res = SimpleMCMC.simpleRWM(model, 10000, 1000, [0.5, 30, 0.01])
res = SimpleMCMC.simpleRWM(model, 101000, 1000)
mean(res[:,2])

res = SimpleMCMC.simpleHMC(model, 10000, 500, [0.5, 0.01, 0.01], 2, 0.001)
res = SimpleMCMC.simpleHMC(model, 1000, 1, 0.1)
mean(res[:,2])

dlmwrite("c:/temp/dump.txt", res)
mean(Weibull(2,1))

###
model = quote
    mu::real
    scale::real

    scale ~ Weibull(2,1)
    mu ~ Uniform(0,1)

    f2 = exp(-dt * scale)
    resid = tx[2:end] - tx[1:end-1] .* f2 - mu * (1.-f2)
    resid ~ Normal(0, 0.01)
end

res = SimpleMCMC.simpleRWM(model, 10000, 500, [0.5, 0.01])
res = SimpleMCMC.simpleRWM(model, 101000, 1000)
mean(res[:,2])

res = SimpleMCMC.simpleHMC(model, 10000, 500, [0.5, 0.01], 5, 0.001)
res = SimpleMCMC.simpleHMC(model, 1000, 1, 0.1)
mean(res[:,2])

dlmwrite("c:/temp/dump.txt", res)


#####################################################
include("../src/SimpleMCMC.jl")

model = quote
    mu::real(4)
    scale::real(3,5)

    resid ~ Normal(0, 0.01)
end

f, nb = SimpleMCMC.buildFunctionWithGradient(model)

###########################################################

f() = 2.0
g(f) = f()
g(f)
# 2.0

f() = 3.0
g(f)
# 2.0
        beta = ones(6)
        x = reshape(__beta[1:6], 3, 2)
        __acc = 1.0
        __tmp_3199 = x[:, 2]
        __tmp_3200 = +(__tmp_3199, z)
        y = sum(__tmp_3200)
        d__acc = 1.0
        dy = zero(y)
        dx = zero(x)
        d__tmp_3199 = zero(__tmp_3199)
        d__tmp_3200 = zero(__tmp_3200)
        d__tmp_3200 += dy
        d__tmp_3199 += if isa(__tmp_3199, Real)
                sum(d__tmp_3200)
            else 
                d__tmp_3200
            end
        dx[:, 2] = d__tmp_3199


SimpleMCMC.buildFunctionWithGradient(model)

__beta = ones(28)
        mu = __beta[1:4]
        beta = reshape(__beta[9:28], 4, 5)
        sigma = __beta[5:8]

        __acc = 0.0
        __tmp_3224 = SimpleMCMC.logpdfNormal(0, 100, mu)
        ____acc_3236 = +(__acc, __tmp_3224)
        __tmp_3225 = SimpleMCMC.logpdfUniform(0, 1000, sigma)
        ____acc_3237 = +(____acc_3236, __tmp_3225)
        __tmp_3226 = *(mu, onerow)
        __tmp_3227 = .^(sigma, 2)
        __tmp_3228 = *(__tmp_3227, onerow)
        __tmp_3229 = SimpleMCMC.logpdfNormal(__tmp_3226, __tmp_3228, beta)
        ____acc_3238 = +(____acc_3237, __tmp_3229)
        __tmp_3230 = beta[ll, :]
        __tmp_3231 = .*(__tmp_3230, X)
        effect = *(__tmp_3231, onecol)
        __tmp_3232 = -(effect)
        __tmp_3233 = exp(__tmp_3232)
        __tmp_3234 = +(1.0, __tmp_3233)
        prob = /(1.0, __tmp_3234)
        __tmp_3235 = SimpleMCMC.logpdfBernoulli(prob, Y)
        ____acc_3239 = +(____acc_3238, __tmp_3235)


        d____acc_3239 = 1.0
        d__tmp_3229 = zero(__tmp_3229)
        d__tmp_3230 = zero(__tmp_3230)
        dbeta = zero(beta)
        d__tmp_3231 = zero(__tmp_3231)
        d__tmp_3225 = zero(__tmp_3225)
        d____acc_3236 = zero(____acc_3236)
        d__tmp_3227 = zero(__tmp_3227)
        d__tmp_3235 = zero(__tmp_3235)
        d__tmp_3232 = zero(__tmp_3232)
        dsigma = zero(sigma)
        deffect = zero(effect)
        d__tmp_3234 = zero(__tmp_3234)
        d__tmp_3233 = zero(__tmp_3233)
        d__tmp_3228 = zero(__tmp_3228)
        d__tmp_3224 = zero(__tmp_3224)
        dmu = zero(mu)
        d____acc_3237 = zero(____acc_3237)
        dprob = zero(prob)
        d____acc_3238 = zero(____acc_3238)
        d__tmp_3226 = zero(__tmp_3226)
        


        d____acc_3238 += if isa(____acc_3238, Real)
                sum(d____acc_3239)
            else 
                d____acc_3239
            end
        d__tmp_3235 += if isa(__tmp_3235, Real)
                sum(d____acc_3239)
            else 
                d____acc_3239
            end
        dprob += .*(begin 
                    tmp = ./(1.0, -(prob, -(1.0, Y)))
                    if isa(prob, Real)
                        sum([tmp])
                    else 
                        tmp
                    end
                end, d__tmp_3235)
        d__tmp_3234 += if isa(__tmp_3234, Real)
                sum([.*(./(-(1.0), .*(__tmp_3234, __tmp_3234)), dprob)])
            else 
                .*(./(-(1.0), .*(__tmp_3234, __tmp_3234)), dprob)
            end
        d__tmp_3233 += if isa(__tmp_3233, Real)
                sum(d__tmp_3234)
            else 
                d__tmp_3234
            end
        d__tmp_3232 += .*(exp(__tmp_3232), d__tmp_3233)
        deffect += -(d__tmp_3232)
        d__tmp_3231 += if isa(__tmp_3231, Real)
                sum([.*(deffect, onecol)])
            else 
                *(deffect, onecol')
            end
        d__tmp_3230 += if isa(__tmp_3230, Real)
                sum([.*(d__tmp_3231, X)])
            else 
                .*(d__tmp_3231, X)
            end
        dbeta[ll, :] = d__tmp_3230
        d____acc_3237 += if isa(____acc_3237, Real)
                sum(d____acc_3238)
            else 
                d____acc_3238
            end
        d__tmp_3229 += if isa(__tmp_3229, Real)
                sum(d____acc_3238)
            else 
                d____acc_3238
            end
        d__tmp_3226 += *(begin 
                    tmp = *([./(-(beta, __tmp_3226), .^(__tmp_3228, 2))], d__tmp_3229)
                    if isa(__tmp_3226, Real)
                        sum(tmp)
                    else 
                        tmp
                    end
                end, d__tmp_3229)
        d__tmp_3228 += *(begin 
                    tmp = *(./(-(./(.^(-(beta, __tmp_3226), 2), .^(__tmp_3228, 2)), 0.0), __tmp_3228), d__tmp_3229)
                    if isa(__tmp_3228, Real)
                        sum(tmp)
                    else 
                        tmp
                    end
                end, d__tmp_3229)
        dbeta += *(begin 
                    tmp = *([./(-(__tmp_3226, beta), .^(__tmp_3228, 2))], d__tmp_3229)
                    if isa(beta, Real)
                        sum(tmp)
                    else 
                        tmp
                    end
                end, d__tmp_3229)

        d__tmp_3227 += if isa(__tmp_3227, Real)
                sum([.*(d__tmp_3228, onerow)])
            else 
                *(d__tmp_3228, onerow')
            end

        dsigma += if isa(sigma, Real)
                sum([.*(.*(2, .^(sigma, -(2, 1))), d__tmp_3227)])
            else 
                .*(.*(2, .^(sigma, -(2, 1))), d__tmp_3227)
            end

        dmu += if isa(mu, Real)
                sum([.*(d__tmp_3226, onerow)])
            else 
                *(d__tmp_3226, onerow')
            end

        d____acc_3236 += if isa(____acc_3236, Real)
                sum(d____acc_3237)
            else 
                d____acc_3237
            end
        d__tmp_3225 += if isa(__tmp_3225, Real)
                sum(d____acc_3237)
            else 
                d____acc_3237
            end
        dsigma += 0.0
        d__tmp_3224 += if isa(__tmp_3224, Real)
                sum(d____acc_3236)
            else 
                d____acc_3236
            end
        dmu += *(begin 
                    tmp = *([./(-(0, mu), .^(100, 2))], d__tmp_3224)
                    if isa(mu, Real)
                        sum(tmp)
                    else 
                        tmp
                    end
                end, d__tmp_3224)
        (____acc_3239, vcat(dmu, dbeta, dsigma))



dy = 1.
x=-1
expre = :((tmp = [(z - x) ./ (1 .^ 2)] * dy ; sum(tmp)) * dy)

eval(expre)