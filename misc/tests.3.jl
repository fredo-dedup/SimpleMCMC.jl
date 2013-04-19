# include("../src/SimpleMCMC.jl")

# using DataFrames

# ds = readdlm("c:/temp/ds.txt")

# taux1 = Float64[ds[:,1]...]
# commit2 = Float64[ds[:,7]...]
# horizon = Float64[ds[:,2]...]
# taux0 = Float64[ds[:,3]...]
# lcommit2 = log(commit2)
# N = length(taux1)

# model = quote
#     tci::real
#     tcc::real
#     taui::real
#     tauc::real

#     tc = tci + tcc .* lcommit2
#     tau = max(1., 1e4taui + 1e4tauc .* lcommit2)

#     taux = max(0, tc - (tc - taux0) .* exp(-horizon ./ tau) )
#     resid = (taux1 - taux) .* commit2
#     resid ~ Normal(0, 1e7)
# end

size(taux1)
size(ds)
ds2 = ds[rand(size(ds,1)) .> 0.95,:]
size(ds2)

taux1 = Float64[ds2[:,1]...]
commit2 = Float64[ds2[:,7]...]
horizon = Float64[ds2[:,2]...]
taux0 = Float64[ds2[:,3]...]
lcommit2 = log(commit2)
N = length(taux1)

# SimpleMCMC.simpleAGD(model, [0.67, -0.032, 0.865, -0.0392])

samp2 = SimpleMCMC.simpleHMC(model, 1000, 0, [0.670, -0.0322, 0.865, -0.04], 2, 1e-5)
# samp2 = SimpleMCMC.simpleNUTS(model, 100, 0, [0.67, -0.034, 0.865, -0.0392])

open("c:/temp/samp2.txt", "w") do io
	println(io, join(keys(samp2.params), '\t'))
	writedlm(io, hcat([s[2] for s in samp2.params]...))
end

