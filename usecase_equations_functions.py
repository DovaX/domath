from domath_core import *



    
"""   
term1=Term(3,1)
term2=Term(5,0)
term3=Term(4,1)
term4=Term(6,0)
term5=Term(7,2)
term6=Term(8,1)
term7=Term(2,2)
term8=Term(-1,3)

c=Expression([term1,term2,term3,term4,term5,term6,term7,term8])
"""

#print(c)
#c.simplify()
#print(c)


termnew=Term(1,2)
termnew2=Term(0,-1)
exp1 = t_plus(termnew,termnew2)
fc1 = Function(exp1)
fc1.plot()


term1=Term(1,2)
term2=Term(-1,1)
term3=Term(1,0)
term2b=Term(1,1)
c=Expression([term1,term2,term3])
d=Expression([term1,term2b,term3])


e=t_multiply(term1,term2)
print(e)
print(c)

f=expr_multiply(c,d)
print(f)



