from Functions import *

#Test element
#This is a square



#A Simple Mesh

#Nodes = [[-1,-1],[0,-1],[1,-1],[-1,0],[0,0],[1,0],[-1,1],[0,1],[1,1]]

##The edges are listed in the direction of the orientation of the tangential vector.
#In the algorithms we require the orientation of the normal so we will take the counterclockwise
#rotation, by pi/2, of the tangential vector as the selected orientation of the normal vector.
#EdgeNodes = [[0,1],[4,1],[8,5],[4,7],[7,8],[6,7],[3,6],[0,3],[5,2],[1,2],[3,4],[4,5]]

#The entries refers to the edges of an element 
    #they are listed in the counterclockwise direction
#ElementEdges  = [[9,8,11,1],[0,1,10,7],[10,3,5,6],[11,2,4,3]]

#Orientations = [[1,-1,-1,1],[1,-1,-1,-1],[1,1,-1,-1],[1,-1,-1,-1]]
                                       
#BoundaryEdges=[0,2,4,5,6,7,8,9]  #A row of the edges that are part of the boundary
#BoundaryNodes=[0,1,2,3,5,6,7,8,9]  #A row of the vertices that are part of the boundary h05.tx

#def test_NodalMatrix():
#Here we will test that the inner product matrix in the nodal space is well-constructed.
Nodes           = [ [-1,-1],[1,-1],[1,1],[-1,1] ]
EdgeNodes       = [ [0,3],[3,2],[2,1],[1,0] ]
Element         = [ 3,2,1,0 ]
Ori             = [ -1,-1,-1,-1 ]
Basis           = [Poly1,Poly2,Poly]
ME,MV,MJ,Edges  = NewLocalMEWEMVWV(J,Basis,Element,EdgeNodes,Nodes,Ori)
	
T1              = np.ones((4,1))
T2              = np.array([x for [x,y] in Nodes])
T3              = np.array([y for [x,y] in Nodes])
print(abs(np.transpose(T1).dot(MV).dot(T1)-4))
print(abs(np.transpose(T1).dot(MV).dot(T2)))
print(abs(np.transpose(T1).dot(MV).dot(T3)))
















