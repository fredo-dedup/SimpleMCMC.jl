include("../src/SimpleMCMC.jl")

# using DataFrames

ds = readdlm("c:/temp/ds.txt")

taux1 = Float64[ds[:,1]...]
commit2 = Float64[ds[:,7]...]
horizon = Float64[ds[:,2]...]
taux0 = Float64[ds[:,3]...]
N = length(taux1)


mu = 0.07
resid = (taux1 - mu) .* commit2
sqrt(dot(resid, resid))

model = quote
    mu::real

    resid = (taux1 - mu) .* commit2
    resid = (dot(resid, resid) / N) ^ .5
    resid ~ Normal(0, 1e7)
end

# llf, np, pmap = SimpleMCMC.buildFunction(model)
# llf([0.077171])

res = SimpleMCMC.simpleAGD(model, [0.5], 10)
sqrt(-res.maximum-log(1e7*sqrt(2pi))) * sqrt(2) * 1e7  # 17.41 M€

############# marche pas
model = quote
    mu::real
    sigma::real

    resid = (taux1 - mu) .* commit2
    resid ~ Normal(0, sigma*1e8)
end

res = SimpleMCMC.simpleAGD(model, [0.2, 0.2], 3)
sqrt(-res.maximum-log(1e7*sqrt(2pi))) * sqrt(2) * 1e7  # 17.41 M€

############### avec plus de params
model = quote
    tc::real
    tau::real

    resid = taux1 - (tc - (tc - taux0) .* exp(-horizon / (1e4*tau))) 
    resid = resid .*commit2
    resid = (dot(resid, resid) / N) ^ .5
    resid ~ Normal(0, 1e7)
end

llf, np, pmap = SimpleMCMC.buildFunction(model)
llf([0.05, 0.07])

res = SimpleMCMC.simpleAGD(model, [0.1, 0.1], 10)

sqrt(-res.maximum-log(1e7*sqrt(2pi))) * sqrt(2) * 1e7  # 14.27 M€

############### avec encore plus de params
lcommit2 = log(commit2)

tci, tcc, taui, tauc = 1., 0., 8000, -364
model = quote
    tci::real
    tcc::real
    taui::real
    tauc::real

    tc = tci + tcc .* lcommit2
    tau = max(1., 1e4taui + 1e4tauc .* lcommit2)

    taux = max(0, tc - (tc - taux0) .* exp(-horizon ./ tau) )
    resid = (taux1 - taux) .* commit2
    resid = (dot(resid, resid) / N) ^ .5
    resid ~ Normal(0, 1e7)
end

llf, np, pmap = SimpleMCMC.buildFunction(model)
llf([1.45, -0.07, 0.8, -0.037])
llf([0.077, 0.,  0.145, 0.])
llf([0.08, 0.,  0.145, 0.])
llf([0.07, 0.,  0.145, 0.])
llf([0.077, 0.,  0.145, 0.])
llf([1.45, -0.07, 0.8, -0.037])

llf, np, pmap = SimpleMCMC.buildFunctionWithGradient(model)
llf([1., -0.1, .1, -0.01])

res = SimpleMCMC.simpleAGD(model, [1., 0, 1., 0], 5)


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

samp = SimpleMCMC.simpleRWM(model, 1000, 100, [1., -0.1, 1., -0.1])


res = SimpleMCMC.simpleAGD(model, [0.1, 0.1], 10)

sqrt(18.104-log(1e7*sqrt(2pi))) * sqrt(2) * 1e7  # 14.27 M€

