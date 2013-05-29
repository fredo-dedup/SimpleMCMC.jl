begin   # ornstein
    srand(1)
    duration = 1000  # 1000 time steps
    mu0 = 10.  # target value
    tau0 = 20  # convergence time
    sigma0 = 0.1  # noise term

    x = fill(NaN, duration)
    x[1] = 1.
    for i in 2:duration
    	x[i] = x[i-1]*exp(-1/tau0) + mu0*(1-exp(-1/tau0)) +  sigma0*randn()
    end
    
    # model definition (note : rescaling on tau and mu)
    model = quote
        mu::real
        itau::real
        sigma::real

        itau ~ Gamma(2, 0.1)
        sigma ~ Gamma(2, 1)
        mu ~ Gamma(2, 1)

        fac = exp(- itau)
        dummy = 13.2
        dummy2 = sum(x)
        # resid = x[2:end] - x[1:end-1] * fac - 10 * mu * (1. - fac)
        resid = x[2:end] - x[1:(end-1)] * fac - 10 * mu * (1. - fac)
        resid ~ Normal(0, sigma)
    end
end

include("../src/SimpleMCMC.jl")

SimpleMCMC.generateModelFunction(model, [1.,.1,1], true, true)
myf, dummy = SimpleMCMC.generateModelFunction(model, [1.,.1,1], true, false)

for i in 1:100
    println(myf([1.,.1,1.]))
end




###############################################################

# module test

# model = quote
#     let 
#         global ll57083
#         local x = Main.x
#         local dummy = 13.2
#         local dummy2 = sum(x)
#         local tmp57030 = x[2:end]
#         local tmp57031 = x[1:-(end,1)]
#         local tmp57063 = .*(2,1)
#         local tmp57070 = .*(2,1)
#         local tmp57077 = .*(2,0.1)
#         function ll57083(__beta::Vector{Float64})
#             try 
#                 local mu = __beta[1]
#                 local itau = __beta[2]
#                 local sigma = __beta[3]
#                 local __acc = 0.0
#                 local d__acc57041 = 1.0
#                 local dtmp57029 = 0.0
#                 local dtmp57027 = 0.0
#                 local dtmp57037 = 0.0
#                 local dtmp57036 = 0.0
#                 local d__acc57038 = 0.0
#                 local dtmp57033 = zeros(Float64,(999,))
#                 local ditau = 0.0
#                 local dtmp57026 = 0.0
#                 local dtmp57035 = 0.0
#                 local d__acc57039 = 0.0
#                 local dsigma = 0.0
#                 local d__acc57040 = 0.0
#                 local dfac = 0.0
#                 local dmu = 0.0
#                 local dtmp57034 = 0.0
#                 local dtmp57032 = zeros(Float64,(999,))
#                 local dtmp57028 = 0.0
#                 local dresid = zeros(Float64,(999,))
#                 local tmp57026 = logpdfGamma(2,0.1,itau)
#                 local __acc57038 = +(__acc,tmp57026)
#                 local tmp57027 = logpdfGamma(2,1,sigma)
#                 local __acc57039 = +(__acc57038,tmp57027)
#                 local tmp57028 = logpdfGamma(2,1,mu)
#                 local __acc57040 = +(__acc57039,tmp57028)
#                 local tmp57029 = -(itau)
#                 local fac = exp(tmp57029)
#                 local tmp57032 = *(tmp57031,fac)
#                 local tmp57033 = -(tmp57030,tmp57032)
#                 local tmp57034 = -(1.0,fac)
#                 local tmp57035 = *(mu,tmp57034)
#                 local tmp57036 = *(10,tmp57035)
#                 local resid = -(tmp57033,tmp57036)
#                 local tmp57037 = logpdfNormal(0,sigma,resid)
#                 local __acc57041 = +(__acc57040,tmp57037)
#                 d__acc57040 = +(d__acc57040,sum(d__acc57041))
#                 dtmp57037 = +(dtmp57037,sum(d__acc57041))
#                 local tmp57043 = -(resid,0)
#                 local tmp57044 = .^(tmp57043,2)
#                 local tmp57045 = .^(sigma,2)
#                 local tmp57046 = ./(tmp57044,tmp57045)
#                 local tmp57047 = -(tmp57046,1.0)
#                 local tmp57048 = ./(tmp57047,sigma)
#                 local tmp57049 = *(tmp57048,dtmp57037)
#                 local tmp57050 = sum(tmp57049)
#                 dsigma = +(dsigma,.*(tmp57050,dtmp57037))
#                 local tmp57051 = -(0,resid)
#                 local tmp57052 = .^(sigma,2)
#                 local tmp57053 = ./(tmp57051,tmp57052)
#                 local tmp57054 = *(tmp57053,dtmp57037)
#                 dresid = +(dresid,.*(tmp57054,dtmp57037))
#                 dtmp57033 = +(dtmp57033,+(dresid))
#                 local tmp57055 = sum(dresid)
#                 dtmp57036 = +(dtmp57036,-(tmp57055))
#                 local tmp57056 = .*(dtmp57036,10)
#                 dtmp57035 = +(dtmp57035,sum(tmp57056))
#                 local tmp57057 = .*(dtmp57035,tmp57034)
#                 dmu = +(dmu,sum(tmp57057))
#                 local tmp57058 = .*(dtmp57035,mu)
#                 dtmp57034 = +(dtmp57034,sum(tmp57058))
#                 local tmp57059 = sum(dtmp57034)
#                 dfac = +(dfac,-(tmp57059))
#                 dtmp57032 = +(dtmp57032,-(dtmp57033))
#                 local tmp57060 = .*(dtmp57032,tmp57031)
#                 dfac = +(dfac,sum(tmp57060))
#                 local tmp57061 = exp(tmp57029)
#                 dtmp57029 = +(dtmp57029,.*(tmp57061,dfac))
#                 ditau = +(ditau,-(dtmp57029))
#                 d__acc57039 = +(d__acc57039,sum(d__acc57040))
#                 dtmp57028 = +(dtmp57028,sum(d__acc57040))
#                 local tmp57062 = +(1,mu)
#                 local tmp57064 = -(tmp57062,tmp57063)
#                 local tmp57065 = -(tmp57064)
#                 local tmp57066 = .*(1,mu)
#                 local tmp57067 = ./(tmp57065,tmp57066)
#                 local tmp57068 = sum(tmp57067)
#                 dmu = +(dmu,.*(tmp57068,dtmp57028))
#                 d__acc57038 = +(d__acc57038,sum(d__acc57039))
#                 dtmp57027 = +(dtmp57027,sum(d__acc57039))
#                 local tmp57069 = +(1,sigma)
#                 local tmp57071 = -(tmp57069,tmp57070)
#                 local tmp57072 = -(tmp57071)
#                 local tmp57073 = .*(1,sigma)
#                 local tmp57074 = ./(tmp57072,tmp57073)
#                 local tmp57075 = sum(tmp57074)
#                 dsigma = +(dsigma,.*(tmp57075,dtmp57027))
#                 dtmp57026 = +(dtmp57026,sum(d__acc57038))
#                 local tmp57076 = +(0.1,itau)
#                 local tmp57078 = -(tmp57076,tmp57077)
#                 local tmp57079 = -(tmp57078)
#                 local tmp57080 = .*(0.1,itau)
#                 local tmp57081 = ./(tmp57079,tmp57080)
#                 local tmp57082 = sum(tmp57081)
#                 ditau = +(ditau,.*(tmp57082,dtmp57026))
#                 (__acc57041,vcat(vec([dmu]),vec([ditau]),vec([dsigma])))
#             catch e
#                 if (e=="give up eval")
#                     return (-(Inf),zero(__beta))
#                 else  
#                     throw(e)
#                 end
#             end
#         end
#     end 
# end

# eval(model)
# logpdfUniform(a,b,x) = (a<x<b) / (b-a)
# logpdfGamma(l,s,x) = exp(-x./s)*l
# logpdfNormal(a,b,x) = dot(x-a, x-a) / b / b

# end

# ll57083 = test.ll57083

# # eval(model)

# for i in 1:100
#     println(ll57083([1.,.1,1.]))
# end

# println(ll57083([1.,1,1.]))

# println(ll57083([1.,.1,1.]))
# println(ll57083([1.,1,1.]))

# println(ll57083([1.,.1,1.]))
# println(ll57083([1.,1,1.]))

# println(ll57083([1.,.1,1.]))
# println(ll57083([1.,1,1.]))

# println(ll57083([1.,.1,1.]))
# println(ll57083([1.,1,1.]))
