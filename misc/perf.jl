require("../src/SimpleMCMC.jl")
using SimpleMCMC


begin
# naming conventions
const ACC_SYM = :__acc
const PARAM_SYM = :__beta
const LLFUNC_NAME = "loglik"
const TEMP_NAME = "tmp"
const DERIV_PREFIX = "d"

	(model2, nparams, pmap) = SimpleMCMC.parseModel(model)
	exparray, finalacc = SimpleMCMC.unfold(model2)
	avars = SimpleMCMC.listVars(exparray, [p.sym for p in pmap])
	dmodel = SimpleMCMC.backwardSweep(exparray, avars)

	body = SimpleMCMC.betaAssign(pmap)
	push!(body, :($ACC_SYM = 0.)) # acc init
	body = vcat(body, exparray)

	push!(body, :($(symbol("$DERIV_PREFIX$finalacc")) = 1.0))
	if contains(avars, finalacc) # remove accumulator, treated above
		delete!(avars, finalacc)
	end
	for v in avars 
		push!(body, :($(symbol("$DERIV_PREFIX$v")) = zero($(symbol("$v")))))
	end

	body = vcat(body, dmodel)

	if length(pmap) == 1
		dn = symbol("$DERIV_PREFIX$(pmap[1].sym)")
		dexp = :(vec([$dn]))  # reshape to transform potential matrices into vectors
	else
		dexp = {:vcat}
		# dexp = vcat(dexp, { (dn = symbol("$DERIV_PREFIX$(p.sym)"); :(vec([$DERIV_PREFIX$(p.sym)])) for p in pmap})
		dexp = vcat(dexp, { :( vec([$(symbol("$DERIV_PREFIX$(p.sym)"))]) ) for p in pmap})
		dexp = expr(:call, dexp)
	end

	push!(body, :(($finalacc, $dexp)))
	func = Main.eval(SimpleMCMC.tryAndFunc(body, true))
end

macro timeit(ex, nit, name)
    @gensym t i
    quote
        $t = Inf
        for $i=1:5
            $t = min($t, @elapsed for j in 1:$nit; $ex; end)
        end
        println(fb, join([$(expr(:quote, name)), iround(time()), JULVERSION, 
            LIBVERSION, MACHINEID, round($t/$nit,6)], "\t"))
        println(join([$(expr(:quote, name)), iround(time()), JULVERSION, 
            LIBVERSION, MACHINEID, round($t/$nit,6)], "\t"))
    end
end



func([1.,0.05,1])

for i in body
	println(i)
end

function noif(__beta)
	
    mu = __beta[1]
        tau = __beta[2]
    sigma = __beta[3]
    __acc = 0.0
    __tmp_683 = SimpleMCMC.logpdfUniform(0,0.1,tau)
    ____acc_696 = +(__acc,__tmp_683)
    __tmp_684 = SimpleMCMC.logpdfUniform(0,2,sigma)
    ____acc_697 = +(____acc_696,__tmp_684)
    __tmp_685 = SimpleMCMC.logpdfUniform(0,2,mu)
    ____acc_698 = +(____acc_697,__tmp_685)
    __tmp_686 = -(0.001)
    __tmp_687 = /(__tmp_686,tau)
    fac = exp(__tmp_687)
    __tmp_688 = x[2:end]
    __tmp_689 = x[1:-(end,1)]
    __tmp_690 = *(__tmp_689,fac)
    __tmp_691 = -(__tmp_688,__tmp_690)
    __tmp_692 = -(1.0,fac)
    __tmp_693 = *(mu,__tmp_692)
    __tmp_694 = *(10,__tmp_693)
    resid = -(__tmp_691,__tmp_694)
    __tmp_695 = SimpleMCMC.logpdfNormal(0,sigma,resid)
    ____acc_699 = +(____acc_698,__tmp_695)
    d____acc_699 = 1.0
    d__tmp_687 = zero(__tmp_687)
    d____acc_698 = zero(____acc_698)
    d__tmp_683 = zero(__tmp_683)
    dsigma = zero(sigma)
    d__tmp_692 = zero(__tmp_692)
    d____acc_696 = zero(____acc_696)
    dmu = zero(mu)
    d____acc_697 = zero(____acc_697)
    d__tmp_690 = zero(__tmp_690)
    d__tmp_684 = zero(__tmp_684)
    d__tmp_694 = zero(__tmp_694)
    d__tmp_685 = zero(__tmp_685)
    d__tmp_693 = zero(__tmp_693)
    d__tmp_691 = zero(__tmp_691)
    dfac = zero(fac)
    dtau = zero(tau)
    dresid = zero(resid)
    d__tmp_695 = zero(__tmp_695)

	d____acc_698 += sum(d____acc_699)
	d__tmp_695 += sum(d____acc_699)
	dsigma += .*(sum( *(./(-(./(.^(-(resid,0),2),.^(sigma,2)),1.0),sigma),d__tmp_695) ),d__tmp_695)
	dresid += .*(*([./(-(0,resid),.^(sigma,2))],d__tmp_695),d__tmp_695)
	d__tmp_691 += dresid
	d__tmp_694 += -(sum(dresid))
	d__tmp_693 += sum([.*(d__tmp_694,10)])
	dmu += sum([.*(d__tmp_693,__tmp_692)])
	d__tmp_692 += sum([.*(d__tmp_693,mu)])
	dfac += -(sum(d__tmp_692))
	d__tmp_690 += -(d__tmp_691)
	dfac += sum([.*(d__tmp_690,__tmp_689)])
	d__tmp_687 += .*(exp(__tmp_687),dfac)
	dtau += sum([.*(./(-(__tmp_686),.*(tau,tau)),d__tmp_687)])
	d____acc_697 += sum(d____acc_698)
	d__tmp_685 += sum(d____acc_698)
	dmu += zero(mu)
	d____acc_696 += sum(d____acc_697)
	d__tmp_684 += sum(d____acc_697)
	dsigma += zero(sigma)
	d__tmp_683 += sum(d____acc_696)
	dtau += zero(tau)

    (____acc_699,vcat(vec([dmu]),vec([dtau]),vec([dsigma])))

end

# +-0  (3), useless sums
function simp1(__beta)
	
    mu = __beta[1]
        tau = __beta[2]
    sigma = __beta[3]
    __acc = 0.0
    __tmp_683 = SimpleMCMC.logpdfUniform(0,0.1,tau)
    ____acc_696 = +(__acc,__tmp_683)
    __tmp_684 = SimpleMCMC.logpdfUniform(0,2,sigma)
    ____acc_697 = +(____acc_696,__tmp_684)
    __tmp_685 = SimpleMCMC.logpdfUniform(0,2,mu)
    ____acc_698 = +(____acc_697,__tmp_685)
    __tmp_686 = -(0.001)
    __tmp_687 = /(__tmp_686,tau)
    fac = exp(__tmp_687)
    __tmp_688 = x[2:end]
    __tmp_689 = x[1:-(end,1)]
    __tmp_690 = *(__tmp_689,fac)
    __tmp_691 = -(__tmp_688,__tmp_690)
    __tmp_692 = -(1.0,fac)
    __tmp_693 = *(mu,__tmp_692)
    __tmp_694 = *(10,__tmp_693)
    resid = -(__tmp_691,__tmp_694)
    __tmp_695 = SimpleMCMC.logpdfNormal(0,sigma,resid)
    ____acc_699 = +(____acc_698,__tmp_695)
    d____acc_699 = 1.0
    d__tmp_687 = zero(__tmp_687)
    d____acc_698 = zero(____acc_698)
    d__tmp_683 = zero(__tmp_683)
    dsigma = zero(sigma)
    d__tmp_692 = zero(__tmp_692)
    d____acc_696 = zero(____acc_696)
    dmu = zero(mu)
    d____acc_697 = zero(____acc_697)
    d__tmp_690 = zero(__tmp_690)
    d__tmp_684 = zero(__tmp_684)
    d__tmp_694 = zero(__tmp_694)
    d__tmp_685 = zero(__tmp_685)
    d__tmp_693 = zero(__tmp_693)
    d__tmp_691 = zero(__tmp_691)
    dfac = zero(fac)
    dtau = zero(tau)
    dresid = zero(resid)
    d__tmp_695 = zero(__tmp_695)

	d____acc_698 += d____acc_699
	d__tmp_695 += d____acc_699
	dsigma += .*(sum( *(./(-(./(.^(resid,2),.^(sigma,2)),1.0),sigma),d__tmp_695) ),d__tmp_695)
	dresid += .*(*([./(resid,.^(sigma,2))],d__tmp_695),d__tmp_695)
	d__tmp_691 += dresid
	d__tmp_694 += -(sum(dresid))
	d__tmp_693 += .*(d__tmp_694,10)
	dmu += .*(d__tmp_693,__tmp_692)
	d__tmp_692 += .*(d__tmp_693,mu)
	dfac += -d__tmp_692
	d__tmp_690 += -d__tmp_691
	dfac += sum([.*(d__tmp_690,__tmp_689)])
	d__tmp_687 += .*(exp(__tmp_687),dfac)
	dtau += .*(./(-(__tmp_686),.*(tau,tau)),d__tmp_687)
	d____acc_697 += d____acc_698
	d__tmp_685 += d____acc_698
	d____acc_696 += d____acc_697
	d__tmp_684 += d____acc_697
	d__tmp_683 += d____acc_696

    (____acc_699,vcat(vec([dmu]),vec([dtau]),vec([dsigma])))

end

function nograd(__beta)
	
    mu = __beta[1]
        tau = __beta[2]
    sigma = __beta[3]
    __acc = 0.0
    __tmp_683 = SimpleMCMC.logpdfUniform(0,0.1,tau)
    ____acc_696 = +(__acc,__tmp_683)
    __tmp_684 = SimpleMCMC.logpdfUniform(0,2,sigma)
    ____acc_697 = +(____acc_696,__tmp_684)
    __tmp_685 = SimpleMCMC.logpdfUniform(0,2,mu)
    ____acc_698 = +(____acc_697,__tmp_685)
    __tmp_686 = -(0.001)
    __tmp_687 = /(__tmp_686,tau)
    fac = exp(__tmp_687)
    __tmp_688 = x[2:end]
    __tmp_689 = x[1:-(end,1)]
    __tmp_690 = *(__tmp_689,fac)
    __tmp_691 = -(__tmp_688,__tmp_690)
    __tmp_692 = -(1.0,fac)
    __tmp_693 = *(mu,__tmp_692)
    __tmp_694 = *(10,__tmp_693)
    resid = -(__tmp_691,__tmp_694)
    __tmp_695 = SimpleMCMC.logpdfNormal(0,sigma,resid)
    ____acc_699 = +(____acc_698,__tmp_695)

end


function nonormal(__beta)
	
    mu = __beta[1]
        tau = __beta[2]
    sigma = __beta[3]
    __acc = 0.0
    __tmp_683 = SimpleMCMC.logpdfUniform(0,0.1,tau)
    ____acc_696 = +(__acc,__tmp_683)
    __tmp_684 = SimpleMCMC.logpdfUniform(0,2,sigma)
    ____acc_697 = +(____acc_696,__tmp_684)
    __tmp_685 = SimpleMCMC.logpdfUniform(0,2,mu)
    ____acc_698 = +(____acc_697,__tmp_685)
    __tmp_686 = -(0.001)
    __tmp_687 = /(__tmp_686,tau)
    fac = exp(__tmp_687)
    __tmp_688 = x[2:end]
    __tmp_689 = x[1:-(end,1)]
    __tmp_690 = *(__tmp_689,fac)
    __tmp_691 = -(__tmp_688,__tmp_690)
    __tmp_692 = -(1.0,fac)
    __tmp_693 = *(mu,__tmp_692)
    __tmp_694 = *(10,__tmp_693)
    resid = -(__tmp_691,__tmp_694)

end


func([1.,0.05,1])
noif([1.,0.05,1])

@elapsed for j in 1:10000; func([1.,0.05,1]); end  # 7.950 7.957 7.981 7.935
@elapsed for j in 1:10000; noif([1.,0.05,1]); end  # 7.968 7.946 7.889 7.984
@elapsed for j in 1:10000; simp1([1.,0.05,1]); end  # 7.748 7.792 7.760 7.770 (-2%)
@elapsed for j in 1:10000; nograd([1.,0.05,1]); end  # 6.237 6.265 6.186 6.243 (-20%)
@elapsed for j in 1:10000; nonormal([1.,0.05,1]); end  # 0.552 0.543 0.547 0.545 (71% on lognormal)


