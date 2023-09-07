#Generalised Logistic Curve

# A: Upper asymptote
# K: Lower asymptote
# C: Typically takes a value of 1. Otherwise, the upper asymptote is A + (K-A) / (C^(1/v))
# B: Growth rate
# v > 0: affects near which asymptote maximum growth occurs
# Q: is related to the value glc(0)
function glc(x; A=-1.0, K=1.0, C=1.0, B=1.0, v=1.0, Q=1.0)
    return A + ( (K-A) / (C + Q*MathConstants.e^(-B*x) )^(1/v) )
end


# for i in -10:10
#    v = i
#    println(v," ", glc(v))
# end

_b = 0.05
_k = 0.01

x = [x/10 for x in -180:180]
y = [glc(i;B=_b, A =-_k, K=_k) for i in x ]
plot(x,y)

max = 1
denom = 10
range = _b*10:max*10

ys = [ [glc(i;B=b/denom) for i in x ] for b in range]
plot(x,ys; title="B, influence in glc", labels= [b/denom for b in range]' )
