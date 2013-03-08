################################

include("../src/SimpleMCMC.jl")

function recap(res)
    print("ess/sec. $(map(iround, res.essBySec)), ")
    print("mean : $(round(mean(res.params[:x]),3)), ")
    println("std : $(round(std(res.params[:x]),3))")
end

model = :(x::real ; x ~ Weibull(1, 1))  # mean 1.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [1.]))  # 3.400 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [1.], 2, 0.8)) # 6.100 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [1.]))  # 400 ess/s

res = SimpleMCMC.simpleNUTS(model, 100000, 1000, [1.])
mean(res.misc[:jmax])  # 3.7
mean(res.misc[:epsilon])  # 3.7

res.misc[:epsilon][1:20]
res.misc[:epsilon][990:1010]


model = :(x::real ; x ~ Weibull(3, 1)) # mean 0.89, std 0.325
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [1.]))  # 6.900 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.6], 2, 0.3)) # 84.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [1.]))  # 22.000 ess/s, correct

model = :(x::real ; x ~ Uniform(0, 2)) # mean 1.0, std 0.577
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [1.]))  # 6.800 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [1.], 1, 0.9)) # 12.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 10000, 1000, [1.]))  # 400 ess/s, very slow due to gradient == 0 ?

res = SimpleMCMC.simpleNUTS(model, 100000, 1000, [1.])
mean(res.misc[:jmax])  # 3.7
mean(res.misc[:epsilon])  # 3.7

res.misc[:epsilon][1:20]
res.misc[:epsilon][990:1010]
res.misc[:jmax][1:20]
res.misc[:jmax][990:1010]



model = :(x::real ; x ~ Normal(0, 1)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.]))  # 16.000 ess/s  7500
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.], 2, 0.8)) # 93.000 ess/s, 49-50.000
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.]))  # 35.000 ess/s, correct, 22-24.0000

model = :(x::real ; x ~ Normal(3, 12)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.]))  # 16.000 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.], 2, 9.)) # 95.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.]))  # 33.000 ess/s, correct

model = :(x::real(10) ; x ~ Normal(3, 12)) # mean 0.0, std 1.0
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, 0.))  # 1.000 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, 0., 3, 7.)) # 30.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, 0.))  # 9.000 ess/s


z = [ rand(100) .< 0.5]
model = :(x::real ; x ~ Uniform(0,1); z ~ Bernoulli(x)) # mean 0.5, std ...
recap(SimpleMCMC.simpleRWM(model, 100000, 1000, [0.5]))  # 5.100 ess/s
recap(SimpleMCMC.simpleHMC(model, 100000, 1000, [0.5], 2, 0.04)) # 10.000 ess/s
recap(SimpleMCMC.simpleNUTS(model, 100000, 1000, [0.5]))  # 6.600 ess/s, correct

myf, n, pmap = SimpleMCMC.buildFunctionWithGradient(model)
mean(z)
myf([0.71])

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

#####################################################################

require("DataFrames")
using DataFrames


SimpleMCMC.calcStats!(res)

################ debuggibg NUTS #####################

    begin 
        init = 1.
        steps = 10
        burnin = 0

        tic() # start timer
        SimpleMCMC.checkSteps(steps, burnin) # check burnin steps consistency
        
        ll_func, nparams, pmap = SimpleMCMC.buildFunctionWithGradient(model) # build function, count the number of parameters
        beta0 = SimpleMCMC.setInit(init, nparams) # build the initial values
        res = SimpleMCMC.setRes(steps, burnin, pmap) #  result structure setup

        # first calc
        llik0, grad0 = ll_func(beta0)
        assert(isfinite(llik0), "Initial values out of model support, try other values")
    end
    # Leapfrog step
    function leapFrog(beta, r, grad, ve, ll)
        local llik

        # println("IN --- beta=$beta, r=$r, grad=$grad, ve=$ve")

        r += grad * ve / 2.
        beta += ve * r
        llik, grad = ll(beta) 
        r += grad * ve / 2.

        # println("OUT --- beta=$beta, r=$r, grad=$grad, llik=$llik")

        return beta, r, llik, grad
    end

    # find initial value for epsilon
    begin
        epsilon = 1.
        jump = randn(nparams)
        beta1, jump1, llik1, grad1 = leapFrog(beta0, jump, grad0, epsilon, ll_func)

        ratio = exp(llik1-dot(jump1, jump1)/2. - (llik0-dot(jump,jump)/2.))
        a = 2*(ratio>0.5)-1.
        while ratio^a > 2^-a
            epsilon = 2^a * epsilon
            beta1, jump1, llik1, grad1 = leapFrog(beta0, jump, grad0, epsilon, ll_func)
            ratio = exp(llik1-dot(jump1, jump1)/2. - (llik0-dot(jump,jump)/2.))
        end
        println("starting epsilon = $epsilon")
    end
    ### adaptation parameters
    const delta = 0.7  # target acceptance
    const nadapt = 1000  # nb of steps to adapt epsilon
    const gam = 0.05
    const kappa = 0.75
    const t0 = 100
    ### adaptation inital values
    hbar = 0.
    mu = log(10*epsilon)
    lebar = 0.0

    # buidtree function
    function buildTree(beta, r, grad, dir, j, ll)
        local beta1, r1, llik1, grad1, n1, s1, alpha1, nalpha1
        local beta2, r2, llik2, grad2, n2, s2, alpha2, nalpha2
        local betam, rm, gradm, betap, rp, gradp
        local dummy, H1
        const deltamax = 100

        if j == 0
            beta1, r1, llik1, grad1 = leapFrog(beta, r, grad, dir*epsilon, ll)
            H1 = llik1 - dot(r1,r1)/2.0
            n1 = ( u_slice <= H1 ) + 0 
            s1 = u_slice < ( deltamax + H1 )

            return beta1, r1, grad1,  
                    copy(beta1), copy(r1), copy(grad1),  
                    copy(beta1), llik1, copy(grad1),  
                    n1, s1, 
                    min(1., exp(H1 - H0)), 1
        else
            betam, rm, gradm,  betap, rp, gradp,  beta1, llik1, grad1,  n1, s1, alpha1, nalpha1 = 
                buildTree(beta, r, grad, dir, j-1, ll)
            if s1 
                if dir == -1
                    betam, rm, gradm,  dummy, dummy, dummy,  beta2, llik2, grad2,  n2, s2, alpha2, nalpha2 = 
                        buildTree(betam, rm, gradm, dir, j-1, ll)
                else
                    dummy, dummy, dummy,  betap, rp, gradp,  beta2, llik2, grad2,  n2, s2, alpha2, nalpha2 = 
                        buildTree(betap, rp, gradp, dir, j-1, ll)
                end
                if rand() < n2/(n2+n1)
                    beta1 = beta2
                    llik1 = llik2
                    grad1 = grad2
                end
                alpha1 += alpha2
                nalpha1 += nalpha2
                s1 = s2 && (dot((betap-betam), rm) >= 0.0) && (dot((betap-betam), rp) >= 0.0)
                n1 += n2
            end

            return betam, rm, gradm,  betap, rp, gradp,  beta1, llik1, grad1,  n1, s1, alpha1, nalpha1
        end
    end

    ### main loop
    for i in 16:16  # i=1
    # i = 8
        # local dummy, alpha, nalpha

        begin
            r0 = randn(nparams)
            H0 = llik0 - dot(r0,r0)/2.

            u_slice  = log(rand()) + H0 # use log ( != paper) to avoid underflow
            
            beta = copy(beta0)
            betap = betam = beta0

            grad = copy(grad0)
            gradp = gradm = grad0

            rp = rm = r0
            llik = llik0

            # inner loop
            j, n = 0, 1
            s = true
        end 

        while s && j < 15
            begin
                dir = (rand() > 0.5) * 2. - 1.
                if dir == -1
                    betam, rm, gradm,  dummy, dummy, dummy,  beta1, llik1, grad1,  n1, s1, alpha, nalpha = 
                        buildTree(betam, rm, gradm, dir, j, ll_func)
                else
                    dummy, dummy, dummy,  betap, rp, gradp,  beta1, llik1, grad1,  n1, s1, alpha, nalpha = 
                        buildTree(betap, rp, gradp, dir, j, ll_func)
                end

                if s1 && rand() < n1/n  # accept and set new beta
                    println("    accepted s1=$s1, n1/n=$(n1/n)")
                    beta = beta1
                    llik = llik1
                    grad = grad1
                    println("==== grad = $grad")
                end

                n += n1
                j += 1
                s = s1 && (dot((betap-betam), rm) >= 0.0) && (dot((betap-betam), rp) >= 0.0)
                println("---  dir=$dir, j=$j, n=$n, s=$s, s1=$s1, alpha/nalpha=$(alpha/nalpha)")

            end
        end 
        
        # epsilon adjustment
        if i <= nadapt  # warming up period
            hbar = hbar * (1-1/(i+t0)) + (delta-alpha/nalpha)/(i+t0)
            le = mu-sqrt(i)/gam*hbar
            lebar = i^(-kappa) * le + (1-i^(-kappa)) * lebar
            epsilon = exp(le)
            println("alpha=$alpha, nalpha=$nalpha, hbar=$hbar, \n le=$le, lebar=$lebar, epsilon=$epsilon")
        else # post warm up, keep same epsilon
            epsilon = exp(lebar)
        end

        # println(llik, beta)
        # i > burnin ? addToRes!(res, pmap, i-burnin, llik, beta != beta0, beta) : nothing

        beta0 = beta ; grad0 = grad ; llik0 = llik
    end



###

type Test2
    v::Vector
end

a = Test2([1,2])

b = a

a
b
b.v[2] = 10

b = Test2(a.v)
b.v[2] = 0

b =deepcopy(a)
a
b
b.v[2] = 20



function test(s::Sample)
    s.beta = [0.0]
end

a = Sample([1.0])
b = Sample([1.0])

a==b
b=a
a==b

test(a)

a


a, b = test(3)

a
b

a[2] = 2
