include("../src/SimpleMCMC.jl")

# using DataFrames

begin
    ds = readdlm("c:/temp/ds.txt")

    taux1 = Float64[ds[:,1]...]
    commit2 = Float64[ds[:,7]...]
    horizon = Float64[ds[:,2]...]
    taux0 = Float64[ds[:,3]...]
    N = length(taux1)


    mu = 0.07
    resid = (taux1 - mu) .* commit2
    sqrt(dot(resid, resid))
end

model = quote
    mu::real

    resid = (taux1 - mu) .* commit2
    resid = (dot(resid, resid) / N) ^ .5
    resid ~ Normal(0, 1e7)
end

# llf, np, pmap = SimpleMCMC.buildFunction(model)
# llf([0.077171])

res = SimpleMCMC.simpleAGD(model, [0.5], 10)  # 0.07717152423519039, 2 iterations, 7.3 sec.
sqrt(-res.maximum-log(1e7*sqrt(2pi))) * sqrt(2) * 1e7  # 17.41 M€

res = SimpleMCMC.simpleNM(model, [0.5], 100) # 0.0771484375, 10 iterations, 4.2 sec.

############### avec plus de params
model = quote
    tc::real
    tau::real

    resid = taux1 - (tc - (tc - taux0) .* exp(-horizon / (1e4*tau))) 
    resid = resid .*commit2
    resid = (dot(resid, resid) / N) ^ .5
    resid ~ Normal(0, 1e7)
end

# llf, np, pmap = SimpleMCMC.buildFunction(model)
# llf([0.05, 0.07])

res = SimpleMCMC.simpleAGD(model, [0.1, 0.1], 10) 
# tau= 0.07449810319275453  tc= 0.052605751042725475  
# 5 iterations, 37.0 sec.
sqrt(-res.maximum-log(1e7*sqrt(2pi))) * sqrt(2) * 1e7  # 14.2739126 M€

res = SimpleMCMC.simpleNM(model, [0.1, 0.1], 100)
# tau= 0.07483534089133173  tc= 0.07477531935510343  
# 25 iterations, 37.4 sec.
sqrt(res.maximum-log(1e7*sqrt(2pi))) * sqrt(2) * 1e7  # 14.273925 M€
dump(res)


############### avec encore plus de params
lcommit2 = log(commit2)

# tci, tcc, taui, tauc = 1., 0., 8000, -364

model = quote
    tci::real
    tcc::real
    taui::real
    tauc::real

    tc = tci + tcc .* lcommit2
    tau = max(1., 1e4taui + 1e4tauc .* lcommit2)

    taux = max(0, tc - (tc - taux0) .* exp(-horizon ./ tau) )
    resid = (taux1 - taux) .* commit2
    resid ~ Normal(0, 1e7)
end

# samp = SimpleMCMC.simpleRWM(model, 1000, 100, [1., -0.1, 1., -0.1])

res = SimpleMCMC.simpleAGD(model, ones(4), 10)
res = SimpleMCMC.simpleAGD(model, [0.1, 0., 0.1, 0.], 10)  # does not converge

res2 = SimpleMCMC.simpleNM(model, [2., 2., 2., 2.], 100)
# taui= -0.2779542259355154  tauc= -0.2893495878879177  tci= -0.30118035577903  tcc= -0.2939220662536732  
# convergence precision not reached, 100 iterations, 230.9 sec.


res = SimpleMCMC.simpleAGD(model, [-0.301, -0.293, -0.278, -0.289], 10) 
sqrt(-res.maximum-log(1e7*sqrt(2pi))) * sqrt(2) * 1e7  # 34429 M€

res = SimpleMCMC.simpleAGD(model, [1., -0.07, 1., -0.03], 10) 
sqrt(-res.maximum-log(1e7*sqrt(2pi))) * sqrt(2) * 1e7  # 34429 M€

res = SimpleMCMC.simpleNM(model, [1., -0.07, 1., -0.03], 100) 
sqrt(-res.maximum-log(1e7*sqrt(2pi))) * sqrt(2) * 1e7  # 34429 M€

sqrt(18.104-log(1e7*sqrt(2pi))) * sqrt(2) * 1e7  # 14.27 M€

