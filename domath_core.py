import cmath
import matplotlib.pyplot as plt
import numpy as np
import math
import random

########## Computing core ####################

class Term:
    def __init__(self,coef,power):
        self.coef=coef
        self.power=power
        
    def __str__(self):
        if self.power == 0:
            return(str(self.coef))
        elif self.power == 1:
            return(str(self.coef)+"x")
        else:
            return(str(self.coef)+"x^"+str(self.power))
    
    def multiply(self,c):
        self.coef=self.coef*c


class Expression:
    def __init__(self,terms):
        self.terms=terms
    
    def __str__(self):
        s=""
        #print(self.terms)
        for i,term in enumerate(self.terms):
            if i == len(self.terms)-1:
                s+=str(term)
            else:
                s+=str(term)+"+"
        #print(s)
        return(s)
    
    def downgrade_to_term(self):
        if len(self.terms)==1:
            new_term=Term(self.terms[0].coef,self.terms[0].power)
            return(new_term)
    
    def add_term(self,term):
        added=False
        for i in range(len(self.terms)):
            if self.terms[i].power==term.power and added == False:
                result=t_plus(self.terms[i],term)
                self.terms[i]=result.downgrade_to_term()
                added=True
        if added == False:
            self.terms.append(term)
        
        self.check_zero_items()
        
    def check_zero_items(self):
        i=0
        while i<len(self.terms):
            if self.terms[i].coef==0 and len(self.terms)>1:
                self.terms.pop(i)
            i+=1
            
    def multiply(self,c):
        for i in range(len(self.terms)):
            self.terms[i].multiply(c)
                
            
    def simplify(self):
        poplist=[]
        for i in range(len(self.terms)):
            for j in range(i+1,len(self.terms)):
                if self.terms[i].power==self.terms[j].power:
                    result=t_plus(self.terms[i],self.terms[j])
                    self.terms[i]=result.downgrade_to_term()
                    if j not in poplist:
                        poplist.append(j)
        poplist.sort(reverse=True)
        for i in range(len(poplist)):
            self.terms.pop(poplist[i])
            


def t_plus(term1,term2):
    
    expr=Expression([])
    if term1.power == term2.power:
        new_term=Term(term1.coef+term2.coef,term1.power)
        expr.terms.append(new_term)
    else:
        expr.terms.append(term1)
        expr.terms.append(term2)
    return(expr)

def t_multiply(term1,term2):
    """multiply two terms"""
    result=Term(term1.coef*term2.coef,term1.power+term2.power)
    return(result)        


def expr_multiply(expr1,expr2):
    result=Expression([])
    for i in range(len(expr1.terms)):
        for j in range(len(expr2.terms)):
            new_term = t_multiply(expr1.terms[i],expr2.terms[j])
            result.add_term(new_term)
    return(result)


        
class Equation:
    def __init__(self,lhs,rhs):
        self.lhs=lhs
        self.rhs=rhs
        
        
    def __str__(self):
        return(str(self.lhs)+"="+str(self.rhs))

    def add(self,term):
        self.lhs.add_term(term)
        self.rhs.add_term(term)
        print(self)
        
    def multiply(self,c):
        self.lhs.multiply(c)
        self.rhs.multiply(c)
        print(self)



class Function:
    def __init__(self,rhs):
        self.rhs=rhs
        

    def __str__(self):
        return("y="+str(self.rhs))
    
    def plot(self):
        x=list(np.arange(-2,2.1,0.01))
        x.pop(200)
        
        y=[]
        def f(x):
            s=0
            for i in range(len(self.rhs.terms)):
                s+=self.rhs.terms[i].coef*(x**self.rhs.terms[i].power)
            return(s)

        for item in x:
            y.append(f(item))        
        
        
        plt.plot(x,y)





class LinearFunction(Function):
    def __init__(self,rhs):
        super().__init__()
    def __str__(self,rhs):
        super().__str__()
    
    





############################ SETS #################################################


class Set:
    def __init__(self):
        pass


class NumberSet(Set):
    #contains only numbers
    def __init__(self,numbers=None):
        self.numbers=set(numbers)
        
    def is_element(self,x):
        if x in self.numbers:
            return(True)
        else:
            return(False)
    
    def sample(self,n):
        set_sample=random.sample(self.numbers,n)
        return(set_sample)        
    

class RealNumbers(NumberSet):
    def __init__(self):
        self.dimension = 1
    
    
    def is_element(self,x):
        if type(x)==float or type(x)==int:
            return(True)
        else:
            return(False)





class NaturalNumbers(NumberSet):
    def __init__(self):
        pass

    def is_element(self,x):
        if (type(x)==float or type(x)==int) and x%1==0:
            return(True)
        else:
            return(False)







class Infinity:
    def __init__(self,sign=1):
        self.sign=sign

    
class Interval(RealNumbers):
    def __init__(self,interval_string):
        left_part = interval_string.split(",")[0]
        right_part = interval_string.split(",")[1]
        if left_part[0]=="(":    
            self.l_cl = False
        else:
            self.l_cl = True
        if right_part[-1]==")":    
            self.r_cl = False
        else:
            self.r_cl = True
            
        if left_part[1:]=="-infty":
            self.left = Infinity(sign=-1)
        else:
            self.left = float(left_part[1:])
            
        if right_part[:-1]=="infty":
            self.right = Infinity(sign=1)
        else:
            self.right = float(right_part[:-1])
        
    
    def is_element(self,x):
        if type(self.right)!=Infinity:
            lower_than_right_bound = x<self.right
        else:
            lower_than_right_bound=True
            
        if type(self.left)!=Infinity:
            higher_than_left_bound = x>self.left
        else:
            higher_than_left_bound=True
        
        
        if (lower_than_right_bound and higher_than_left_bound) or (self.l_cl and x==self.left) or (self.r_cl and x==self.right):
            return(super().is_element(x))
        else:
            return(False)
        
    def bounded_sample(self,n):
        if type(self.right)==Infinity or type(self.left)==Infinity:
            print("Not implemented yet for unbounded intervals")
            return([])
        else:
            sample = []
            for _ in range(n):
                sample.append(self.left + (self.right-self.left)*random.random())
            return(sample)
        
        
    @classmethod 
    def from_list_of_numbers(self,numbers,l_cl="[",r_cl="]"):
        #Convex hull of a set of numbers
        minimum = min(numbers)
        maximum = max(numbers)
        interval_string = l_cl+str(minimum)+","+str(maximum)+r_cl
        return(Interval(interval_string))

class Point:
    def __init__(self,coor):
        self.coor=coor
    def add_vector(self,vector):
        new_coor=[vector.components[i]+self.coor[i] for i,x in enumerate(vector.components)]  
        new_point=Point(new_coor)
        return(new_point)

class Vector:
    def __init__(self,components):
        self.components=components #list

    @classmethod
    def from_points(self,point1,point2):
        print(point1.coor)
        print(point2.coor)
        components = [point2.coor[i]-x for i,x in enumerate(point1.coor)]        
        return(Vector(components))

    def multiply(self,coef):
        new_components=[x*coef for x in self.components]
        return(Vector(new_components))


def cartesian_product(vector_spaces):
    result=VectorSpace.from_vector_spaces(vector_spaces)
    return(result)



class Line:
    def __init__(self,point1,point2):
        assert len(point1.coor) == len(point2.coor)
        self.point1 = point1 #A
        self.point2 = point2 #B
        self.vector = Vector.from_points(self.point1,self.point2) #B-A
        
    @classmethod 
    def from_vector(self,point1,vector):
        coor2 = [point1.coor[i]+x for i,x in enumerate(vector.components)]    
        return(Line(point1,Point(coor2)))
        
    def is_element(self,point):
        is_point1=sum([point.coor[i]!=self.point1.coor[i] for i in range(len(point.coor))]) #same point -> 0
        is_point2=sum([point.coor[i]!=self.point2.coor[i] for i in range(len(point.coor))]) #same point -> 0
        if is_point1==0 or is_point2==0:
            return(True)
        
        vector1 = self.vector
        vector2 = Vector.from_points(point,self.point1)
        non_zero_component_indices = [vector2.components.index(point) for point in vector2.components if point!=0]
        assert len(non_zero_component_indices)>0
        index = non_zero_component_indices[0]
        norm_vector2 = [point/vector2.components[index] for point in vector2.components]
        norm_vector1 = [point/vector1.components[index] for point in vector1.components]
        is_same_vector=sum([norm_vector1[i]!=norm_vector2[i] for i in range(len(norm_vector1))]) #same vector -> 0
        if is_same_vector==0:
            return(True)
        return(False)
    
    def evaluate(self,coor_index,x):
        x_distance=x-self.point1.coor[coor_index]
        vector_component = self.vector.components[coor_index]
        scaled_vector=self.vector.multiply(x_distance/vector_component)
        new_point = self.point1.add_vector(scaled_vector)
        return(new_point)
    
    def sample_from_interval(self,n,interval,coor_index):
        numbers=interval.bounded_sample(n)
        
        sample=[self.evaluate(coor_index,x) for x in numbers] 
        return(sample)
  
 

class VectorSpace:
    def __init__(self,point=None,vectors=[]):
        self.point=point
        self.vectors = vectors
        self.dimensions = len(vectors)
        
    def is_element(self,x):
        assert len(x.coor)==len(self.point.coor)
        if self.dimensions == 1:
            line = Line.from_vector(self.point,self.vectors[0])
            return(line.is_element(x))
        else:
            print("Not implemented yet, other dimension than 1") # TODO
            return(False)
    
    def sample_from_interval(self,n,interval,coor_index):
        if self.dimensions == 1:
            line = Line.from_vector(self.point,self.vectors[0])
            sample=line.sample_from_interval(n,interval,coor_index)
           
            
            return(sample)
        else:
            print("Not implemented yet, other dimension than 1") # TODO            
     
    
    @classmethod
    def from_vector_spaces(self,spaces):
        #what if intersection - todo - assumes disjoint spaces, missing return VectorSpace!!!
        self.dimension = sum([x.dimension for x in spaces])



#Map
class Map:
    def __init__(self,from_set,to_set,function):
        self.from_set=from_set
        self.to_set=to_set
        self.function=function
        self.norm=self.is_norm()
        
    def evaluate(self,point):
        assert self.from_set.is_element(point)
        
        return(self.function(point))
        
    def map_points(self,points):
        for i,point in enumerate(points):
            points[i]=self.evaluate(point)
        return(points)
    
    def is_norm(self):
        
        return(True)






def euclidean_norm(point):
    result=math.pow(sum([x*x for x in point.coor]),1/len(point.coor))
    return(result)














############################ LOGIC ###########################################
class Proposition:
    def __init__(self,rule):
        

class Implication:
    def __init__(self,assumptions,result):
        
        pass


class Equivalence:
    def __init__(self,assumptions,result):
        pass
    
    
class Theorem:
    def __init__(self):
        pass

class Axiom:
    def __init__(self):
        pass





############################### FUNCTION SPACES ###################################


class ContinuousF(Function):
    def __init__(self):
        pass


############################### Complex numbers ###################################   


def complex_power(z,n):
    result=1
    for i in range(n):
        result=result*z
    return(result)
    

def evaluate_complex_expression(value,expr):
    result=0
    for term in expr.terms:
        result+=term.coef*complex_power(value,term.power)
    return(result)


def find_unit_circle_roots(expr,numbers):
    tol=0.000001
    bound=np.prod(numbers)
    solutions=[]
    for i in range(1,bound):
        value=cmath.rect(1,cmath.pi/(bound/2)*i)
        evaluation=evaluate_complex_expression(value,expr)
        #print(evaluation)
        if abs(evaluation)<tol:
            solutions.append(i)
        #print()
        
        
    return(solutions)




