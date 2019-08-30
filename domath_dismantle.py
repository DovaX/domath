"""DISMANTLING PART - STRING TO EXPRESSION"""
from domath_equation import * 
def dismantle_term(string):
    
    list1=string.split('x')
    coef=int(list1[0]+'1' if (list1[0]=='-' or list1[0]=='') else list1[0])
    if len(list1)==2:
        if list1[1]=='':
            power=1
        else:
            power=int(list1[1].replace('^',''))
    else:
        power=0
    term=Term(coef,power)
    return(term)

def dismantle_expression(string):
    expr=Expression([])
    string=string.replace('-','+-')
    list1=string.split('+')
    #list2=[item for sublist in [x.split('-') for x in list1] for item in sublist]
    
    for term_string in list1:
        term=dismantle_term(term_string)
        expr.add_term(term)
        
    
    return(expr)
    
 
"""Usage:"""    

#dismantle_expression('1x+2-3x^2+5x')


#dismantle_term('-5x^10')