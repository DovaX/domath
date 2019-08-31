from domath_core import *
#Usecase of the domath_core (Sets and Vector space part)

R = RealNumbers()
N = NaturalNumbers()

A=NumberSet([1,2,3])


I = Interval("<5.5,7.1)")


R0plus = Interval("<0,infty)")            
            
            
            
print(R0plus.is_element(1.0))
print(A.sample(1))


R2 = cartesian_product([R,R])

  
line1=Line(Point([1,1]),Point([4,3]))
print(line1.is_element(Point([10,7])))
     




print(euclidean_norm(Point([1,1])))



X = VectorSpace(Point([1,1]),[Vector([1,2])])

print(X.is_element(Point([2,3])))


norm = Map(X,R,euclidean_norm)

print([x.coor for x in X.sample_from_interval(5,I,0)])

value=norm.evaluate(Point([2,3]))
print(value)

Y = VectorSpace()


sample=I.bounded_sample(10000)
I2 = Interval.from_list_of_numbers(sample)
print(I2.left,I2.right)

points=X.sample_from_interval(10,I,0)
print("Points",[points[i].coor for i,x in enumerate(points)])
results=norm.map_points(points)
print(results)