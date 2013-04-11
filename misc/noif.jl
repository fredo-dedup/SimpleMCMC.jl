##### raw
    :(mu = __beta[1])
    :(tau = __beta[2])
    :(sigma = __beta[3])
    :(__acc = 0.0)
    :(##tmp#683 = SimpleMCMC.logpdfUniform(0,0.1,tau))
    :(##__acc#696 = +(__acc,##tmp#683))
    :(##tmp#684 = SimpleMCMC.logpdfUniform(0,2,sigma))
    :(##__acc#697 = +(##__acc#696,##tmp#684))
    :(##tmp#685 = SimpleMCMC.logpdfUniform(0,2,mu))
    :(##__acc#698 = +(##__acc#697,##tmp#685))
    :(##tmp#686 = -(0.001))
    :(##tmp#687 = /(##tmp#686,tau))
    :(fac = exp(##tmp#687))
    :(##tmp#688 = x[2:end])
    :(##tmp#689 = x[1:-(end,1)])
    :(##tmp#690 = *(##tmp#689,fac))
    :(##tmp#691 = -(##tmp#688,##tmp#690))
    :(##tmp#692 = -(1.0,fac))
    :(##tmp#693 = *(mu,##tmp#692))
    :(##tmp#694 = *(10,##tmp#693))
    :(resid = -(##tmp#691,##tmp#694))
    :(##tmp#695 = SimpleMCMC.logpdfNormal(0,sigma,resid))
    :(##__acc#699 = +(##__acc#698,##tmp#695))
    :(d##__acc#699 = 1.0)
    :(d##tmp#687 = zero(##tmp#687))
    :(d##__acc#698 = zero(##__acc#698))
    :(d##tmp#683 = zero(##tmp#683))
    :(dsigma = zero(sigma))
    :(d##tmp#692 = zero(##tmp#692))
    :(d##__acc#696 = zero(##__acc#696))
    :(dmu = zero(mu))
    :(d##__acc#697 = zero(##__acc#697))
    :(d##tmp#690 = zero(##tmp#690))
    :(d##tmp#684 = zero(##tmp#684))
    :(d##tmp#694 = zero(##tmp#694))
    :(d##tmp#685 = zero(##tmp#685))
    :(d##tmp#693 = zero(##tmp#693))
    :(d##tmp#691 = zero(##tmp#691))
    :(dfac = zero(fac))
    :(dtau = zero(tau))
    :(dresid = zero(resid))
    :(d##tmp#695 = zero(##tmp#695))
    :(d##__acc#698 += if isa(##__acc#698,Real)
                sum(d##__acc#699)
            else 
                d##__acc#699
            end)
    :(d##tmp#695 += if isa(##tmp#695,Real)
                sum(d##__acc#699)
            else 
                d##__acc#699
            end)
    :(dsigma += .*(begin 
                    tmp = *(./(-(./(.^(-(resid,0),2),.^(sigma,2)),1.0),sigma),d##tmp#695)
                    if isa(sigma,Real)
                        sum(tmp)
                    else 
                        tmp
                    end
                end,d##tmp#695))
    :(dresid += .*(begin 
                    tmp = *([./(-(0,resid),.^(sigma,2))],d##tmp#695)
                    if isa(resid,Real)
                        sum(tmp)
                    else 
                        tmp
                    end
                end,d##tmp#695))
    :(d##tmp#691 += if isa(##tmp#691,Real)
                sum(dresid)
            else 
                dresid
            end)
    :(d##tmp#694 += if isa(##tmp#694,Real)
                -(sum(dresid))
            else 
                -(dresid)
            end)
    :(d##tmp#693 += if isa(##tmp#693,Real)
                sum([.*(d##tmp#694,10)])
            else 
                *(transpose(10),d##tmp#694)
            end)
    :(dmu += if isa(mu,Real)
                sum([.*(d##tmp#693,##tmp#692)])
            else 
                *(d##tmp#693,transpose(##tmp#692))
            end)
    :(d##tmp#692 += if isa(##tmp#692,Real)
                sum([.*(d##tmp#693,mu)])
            else 
                *(transpose(mu),d##tmp#693)
            end)
    :(dfac += if isa(fac,Real)
                -(sum(d##tmp#692))
            else 
                -(d##tmp#692)
            end)
    :(d##tmp#690 += if isa(##tmp#690,Real)
                -(sum(d##tmp#691))
            else 
                -(d##tmp#691)
            end)
    :(dfac += if isa(fac,Real)
                sum([.*(d##tmp#690,##tmp#689)])
            else 
                *(transpose(##tmp#689),d##tmp#690)
            end)
    :(d##tmp#687 += .*(exp(##tmp#687),dfac))
    :(dtau += if isa(tau,Real)
                sum([.*(./(-(##tmp#686),.*(tau,tau)),d##tmp#687)])
            else 
                .*(./(-(##tmp#686),.*(tau,tau)),d##tmp#687)
            end)
    :(d##__acc#697 += if isa(##__acc#697,Real)
                sum(d##__acc#698)
            else 
                d##__acc#698
            end)
    :(d##tmp#685 += if isa(##tmp#685,Real)
                sum(d##__acc#698)
            else 
                d##__acc#698
            end)
    :(dmu += zero(mu))
    :(d##__acc#696 += if isa(##__acc#696,Real)
                sum(d##__acc#697)
            else 
                d##__acc#697
            end)
    :(d##tmp#684 += if isa(##tmp#684,Real)
                sum(d##__acc#697)
            else 
                d##__acc#697
            end)
    :(dsigma += zero(sigma))
    :(d##tmp#683 += if isa(##tmp#683,Real)
                sum(d##__acc#696)
            else 
                d##__acc#696
            end)
    :(dtau += zero(tau))
    :(##__acc#699,vcat(vec([dmu]),vec([dtau]),vec([dsigma])))

###########################################################################""



__beta = [1., 0.1, 1]

begin
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
end

d____acc_698 += sum(d____acc_699)
d__tmp_695 += sum(d____acc_699)
dsigma += .*(begin 
                tmp = *(./(-(./(.^(-(resid,0),2),.^(sigma,2)),1.0),sigma),d__tmp_695)
                
                    sum(tmp)
                
            end,d__tmp_695)
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

