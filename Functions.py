from numpy import pi
import numpy as np
import math
#from sympy import Matrix
import pylab
#import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#from scipy.interpolate import Rbf
import pickle
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

#Data 
#Here the forcing diffusion coefficient, forcing terms, initial conditions, Dirichlet and Neumann conditions of the
#problem:
#Find E,B such that
#B_t=-curl E
#E=-nu curl B

def DiffusionCoeff(x,y):
    #This is the diffusion coefficient
    #this function is scalar valued
    return 1

def EssentialBoundaryCond(x,y,t):
    #These are the essecial boundary conditions
    #return math.exp(t+x)-math.exp(t+y) #Solution#1 does not include J cross B term
    #return 1 #Solution #2 does not include J cross B term but is linear
    #return -math.exp(y+t) #Solution #3 it includes J cross B term
    #return math.exp(t+x)-math.exp(t+y)#+math.exp(t) #Solution 4 includes J cross B term
    #return 20*math.exp(t+x)-20*math.exp(t+y)-x*y*math.exp(t) #Solution 4 Includes J cross B term
    #return ( 50*(math.exp(x)-math.exp(y))+math.cos(x*y)+math.sin(x*y) )*math.exp(t) #Solution 5 includ
    return -( 50*(math.exp(x)-math.exp(y))+math.cos(x*y)+math.sin(x*y) )*math.exp(-t)     



def InitialCond(x,y):
    #These are the initial condition on the magnetic field
    #r must be a 2 dimensional array
    #this function is vector valued
    #It must be divergence free
    
    #return math.exp(y),math.exp(x) #Solution #1 does not include J cross B term
    #return 2*y,3*x #Solution #1 does not include J cross B term but is linear
    #return math.exp(y),0 #Solution #3 it includes J cross B term
    #return math.exp(y),math.exp(x) #Solution#4  includes J cross B term
    #return 20*math.exp(y)+x,20*math.exp(x)-y#Solution 5 Includes J cross B term
    
#    Bx = 50*math.exp(y)+x*math.sin(x*y)-x*math.cos(x*y)
#    By = 50*math.exp(x)-y*math.sin(x*y)+y*math.cos(x*y)
#    return Bx,By #Solution 6 includes JxB
    Bx = 50*math.exp(y)+x*math.sin(x*y)-x*math.cos(x*y)
    By = 50*math.exp(x)-y*math.sin(x*y)+y*math.cos(x*y)
    return Bx,By #Solution 6 includes JxB



def ExactB(x,y,t):
    #This is the exact Magnetic field
    #Must be divergence free
    #return math.exp(t+y),math.exp(t+x) #Solution #1 does not include J cross B term
    #return 2*y,3*x #Solution #2 does not include J cross B term but is linear
    #return math.exp(y+t),0 #Solution #3 it includes J cross B term
    #return math.exp(t+y),math.exp(t+x) #Solution#4  includes J cross B term
    #return 20*math.exp(y+t)+x*math.exp(t),20*math.exp(x+t)-y*math.exp(t)#Solution 5 Includes J cross B term
    
    Bx = ( 50*math.exp(y)+x*math.sin(x*y)-x*math.cos(x*y) )*math.exp(-t)
    By = ( 50*math.exp(x)-y*math.sin(x*y)+y*math.cos(x*y) )*math.exp(-t)
    return Bx,By#Solution 6 includes JxB

def ExactE(x,y,t):
    #This is the exact Electric field
    #return math.exp(t+x)-math.exp(t+y) #Solution #1 does not include J cross B term
    #return 1 #Solution #2 does not include J cross B term but is linear
    #return -math.exp(y+t) #Solution#3  includes J cross B term
    #return math.exp(t+x)-math.exp(t+y)+math.exp(t) #Solution 4 Includes J cross B term
    #return 20*math.exp(t+x)-20*math.exp(t+y)-x*y*math.exp(t)#Solution 5 Includes J cross B term
#   return ( 50*( math.exp(x)-math.exp(y) )+math.cos(x*y)+math.sin(x*y) )*math.exp(t) #Solution 6 includes JxB
    return -( 50*(math.exp(x)-math.exp(y))+math.cos(x*y)+math.sin(x*y) )*math.exp(-t)   

def J(x,y):
    #return 0,0 #for solutions that do not inclide J x B
    #return math.exp(y),0 #Solution#3  includes J cross B term
    #return 2*math.exp(-x),math.exp(-y) #Solution#4  includes J cross B term
    #return 0,03s
    #return -(x*y)/(40*math.exp(x)-2*y),(x*y)/(40*math.exp(y)+2*x) #solution 5 includes JxB
    
#    Jx = ( (x**2+y**2+1)*(math.sin(x*y)+math.cos(x*y)) )/( 2*(50*math.exp(x)-y*math.sin(x*y)+y*math.cos(x*y)) )
#    Jy = -( (x**2+y**2+1)*(math.sin(x*y)+math.cos(x*y)) )/( 2*(50*math.exp(y)+x*math.sin(x*y)-x*math.cos(x*y)) )
#    return Jx,Jy #Solution 6 includes JxB
    Jx = ( (x**2+y**2-1)*(math.sin(x*y)+math.cos(x*y))-100*math.exp(x)+100*math.exp(y) )/( 2*(50*math.exp(x)-y*math.sin(x*y)+y*math.cos(x*y)) )
    Jy = -( (x**2+y**2-1)*(math.sin(x*y)+math.cos(x*y))-100*math.exp(x)+100*math.exp(y) )/( 2*(50*math.exp(y)+x*math.sin(x*y)-x*math.cos(x*y)) )
    return Jx,Jy #Solution 6 includes JxB



def Poly1(x,y):
    return 1,0
def Poly2(x,y):
    return 0,1
def Poly3(x,y):
    return x,0
def Poly4(x,y):
    return y,0
def Poly5(x,y):
    return 0,x
def Poly6(x,y):
    return 0,y
def Poly(x,y):
    return x,y
















#MeshRetrievalFunctions


def GetMesh(file):
    #This function will, provided a text file in the format of meshes in Mathematica,
    #return the coordinates of the nodes, the Edge Nodes and the Nodes of each element
    UnprocessedMesh=open(file).read()
    FirstCut=UnprocessedMesh.split('}}')
    Nodes=FirstCut[0]+'}'
    EdgeNodes,Elements,trash=FirstCut[1].split(']}')
    for rep in (('{','['),('}',']'),('*^','*10**'),('Line[',''),('Polygon[',''),(']]',']')):
        Nodes=Nodes.replace(rep[0],rep[1])
        EdgeNodes=EdgeNodes.replace(rep[0],rep[1])  #Replace the characters in the first position of the
        Elements=Elements.replace(rep[0],rep[1])     #parenthesis with the second character
                                                    

    Nodes=Nodes+']'
    EdgeNodes=EdgeNodes+']'
    Elements=Elements+']' #add the last ]
    Nodes=eval(Nodes)
    EdgeNodes=eval(EdgeNodes)# turn string to list
    Elements=eval(Elements)
    EdgeNodes=[(np.array(y)-1).tolist() for y in EdgeNodes]
    Elements=[(np.array(y)-1).tolist() for y in Elements]
    #EdgeNodes=np.array(EdgeNodes)-1
    #Elements=np.array(Elements)-1 #subtract one from each position in the array(lists in mathematica begin with 1)
    return Nodes,EdgeNodes,Elements


#def EdgesElement(EdgeNodes,Elements):
    #This function will return the Edges of an element provided a list of the Nodes of the edges 
    #and the nodes of the elements
#    NumberElements=len(Elements)
#    NumberEdges=len(EdgeNodes)
#    ElementEdges=[0]*NumberElements
#    for i in range(NumberElements): #loop over elements
#        NumberNodesEdges=len(Elements[i]) #keep in mind there are the same number of Edges as there a
                                     #of nodes
#        ElementEdge=[0]*NumberNodesEdges    
#        Element=[0]*(NumberNodesEdges+1)
#        Element[0:NumberNodesEdges]=Elements[i] #pick a particular element
        
#        Element[NumberNodesEdges]=Element[0] # add the first vertex as the last vertex (so that edges become every pair)
#        for j in range(NumberNodesEdges): #run over every pair of vertices(edges)
#            Edge=[Element[j],Element[j+1]]
#            for ell in range(NumberEdges): #run over all edges
#                if Edge==EdgeNodes[ell] or [Edge[1],Edge[0]]==EdgeNodes[ell]: #if an the edge agrees, in any direction,
#                    ElementEdge[j]=ell #with one in the list of edges then we have
#        
#        ElementEdges[i]=ElementEdge
#    return ElementEdges                                                       #identified the edge

def EdgesElement(EdgeNodes,Elements):
    #This function will return the Edges of an element provided a list of the Nodes of the edges 
    #and the nodes of the elements
    NumberElements = len(Elements)
    NumberEdges = len(EdgeNodes)
    ElementEdges = [0]*NumberElements
    for i in range(NumberElements): #loop over elements
        NumberNodesEdges = len(Elements[i]) #keep in mind there are the same number of Edges as there a
                                     #of nodes
        ElementEdge = [0]*NumberNodesEdges    
        Element = [0]*(NumberNodesEdges+1)
        Element[0:NumberNodesEdges] = Elements[i] #pick a particular element
        
        Element[NumberNodesEdges] = Element[0] # add the first vertex as the last vertex (so that edges become every pair)
        for j in range(NumberNodesEdges): #run over every pair of vertices(edges)
            Edge = [Element[j],Element[j+1]]
            if Edge in EdgeNodes:
                ElementEdge[j] = EdgeNodes.index(Edge)
            else:
                Edge = [Element[j+1],Element[j]]
                ElementEdge[j] = EdgeNodes.index(Edge)
        ElementEdges[i]=ElementEdge
    return ElementEdges  


   
def Boundary(Nodes):
    #Given an array of Nodes this routine gives the nodes that lie on the boundary
    NumberNodes=len(Nodes)
    BoundaryNodes=[-1]*NumberNodes
    NumberBoundaryNodes=0
    for i in range(NumberNodes):
        Node=Nodes[i]
        if abs(Node[0]-1)<10**-10 or abs(Node[0]+1)<10**-10 or abs(Node[1]-1)<10**-10 or abs(Node[1]+1)<10**-10:
            BoundaryNodes[NumberBoundaryNodes]=i
            NumberBoundaryNodes=NumberBoundaryNodes+1
    return BoundaryNodes[0:NumberBoundaryNodes]

def Mesh(file):
    #Provided a file with a mesh in the language of mathematica this routine will return
    #four lists, Nodes, EdgeNodes,ElementEdges,BoundaryNodes
    Nodes,EdgeNodes,Elements=GetMesh(file) 
    ElementEdges=EdgesElement(EdgeNodes,Elements)
    BoundaryNodes=Boundary(Nodes)
    
    return Nodes,EdgeNodes,ElementEdges,BoundaryNodes

def FindVertecesEdges(Nodes,EdgeNodes):
    #This function, given a set of Edges, will return an array
    #the ith element of this array is the set of all edges that
    #have the ith vertex as an edpoint
    NumberNodes=len(Nodes)
    VertecesEdges=[[]]*NumberNodes
    i=0
    for Edge in EdgeNodes:
        v1=Edge[0]
        v2=Edge[1]
        VertecesEdges[v1]=list(set().union(VertecesEdges[v1],[i]))
        VertecesEdges[v2]=list(set().union(VertecesEdges[v2],[i]))
        i=i+1
    return VertecesEdges




def ProcessedMesh(Pfile):
    with open(Pfile, "rb") as fp:   # Unpickling
        N,E,EE,B,O= pickle.load(fp)
    return N,E,EE,B,O

























#AuxiliaryFunctions

def ElementCoordinates(Element,EdgeNodes,Nodes):
    #provided an element of a mesh this function returns its vertices 
    #as an array of the dimension of the number of edges on the element
    N=len(Element)
    Vertices=[0]*N #The number of vertices agrees with the number of edges
    Edge=Element[0]
    Vertices[0]=Nodes[EdgeNodes[Edge][0]]
    Vertices[1]=Nodes[EdgeNodes[Edge][1]] #The first two vertices are those in the first edge
    for i in range(1,N-1):
        Edge=EdgeNodes[Element[i]]
        v1=Nodes[Edge[0]]
        v2=Nodes[Edge[1]]    #new vertex added is the one on the i+1 edge that is not in the ith edge
        if Vertices[i-1]==v1:
            Vertices[i+1]=v2
        elif Vertices[i-1]==v2:
            Vertices[i+1]=v1
        elif Vertices[i]==v1:
            Vertices[i+1]=v2
        else:
            Vertices[i+1]=v1
    return Vertices

def Orientation(Element,EdgeNodes,Nodes):
    #Provided an element of a mesh this function returns a vector of dimension
    #as large as the number of edges with the orientation of the normal vector
    #1 if the orientation is outward and -1 if it is inward.
    #This algorithm assumes that the element is convex
    
    N=len(Element)
    #first we need to find a point inside the element to give indication of the orientation
    Vertices=ElementCoordinates(Element,EdgeNodes,Nodes)
    xinside=0
    yinside=0
    for i in range(N):
        xinside=xinside+Vertices[i][0]
        yinside=yinside+Vertices[i][1]
    xinside=xinside/N
    yinside=yinside/N   #since the element is convex any convex combination of the vertices lies inside
    
    #Now we move on to finding the orientation of the edges
    Ori=[0]*(N+1)
    
    for i in range(N):
        Edge=EdgeNodes[Element[i]]
        [x1,y1]=Nodes[Edge[0]]
        [x2,y2]=Nodes[Edge[1]]
        #This is the inner product between n, the vector t=(x2,y2)-(x1,y1) rotated pi/2 counterclockwise
        # and the vector u=(xinside,yinside)-(x1,y1) which should point towards the interior of the element
        #if the result is negative it means that the angle between t and u is larger than 90 which implies that
        #n points outside of the element
        sign=(y2-y1)*(xinside-x1)+(x1-x2)*(yinside-y1) 
        if sign<0:
            Ori[i]=1
        else: 
            Ori[i]=-1
    Ori[N]=Ori[0]
    return Ori

def StandardElement(Element,EdgeNodes,Nodes,Ori):
    #This routine will reorient, if necessary, the edges of the element to agree with stokes theorem,
    #This is to say that the edges will be reoriented in such a way that the element will be traversed in the
    #Counterclockwise direction and rotation by pi/2 in the counterclockwise direction of the tangential vector
    #will result in an outward normal vector.
    #The last vertex,edge will be the first. This is in order to complete the loop.
    N                = len(Element)
    OrientedEdges    = [0]*(N+1)
    OrientedVertices = [0]*(N+1)
    
    for i in range(N):
        if Ori[i]==1:
            OrientedEdges[i] = EdgeNodes[Element[i]] #If they are "well-oriented" then do not alter them
        else:
            [v1,v2]          = EdgeNodes[Element[i]] #Otherwise reverse the order of their vertices
            OrientedEdges[i] = [v2,v1]

        OrientedVertices[i]  = Nodes[OrientedEdges[i][0]]
    OrientedEdges[N]    = OrientedEdges[0]
    OrientedVertices[N] = OrientedVertices[0]
    return OrientedVertices,OrientedEdges


def Centroid(Element,EdgeNodes,Nodes,Ori):
    #This function, when provided with an element, will return its centroid or barycenter.
    N=len(Element)
    Cx=0
    Cy=0
    A=0
    Vertices,Edges = StandardElement(Element,EdgeNodes,Nodes,Ori)
    for i in range(N):
        xi=Vertices[i][0]
        yi=Vertices[i][1]
        xiplusone=Vertices[i+1][0]
        yiplusone=Vertices[i+1][1]
        Cx=Cx+(xi+xiplusone)*(xi*yiplusone-xiplusone*yi) #This formula is in Wikipedia
        Cy=Cy+(yi+yiplusone)*(xi*yiplusone-xiplusone*yi)
        A=A+xi*yiplusone-xiplusone*yi
    A=0.5*A
    Cx=Cx/(6*A)
    Cy=Cy/(6*A)
    return Cx,Cy,A,Vertices,Edges

def InternalObjects(Boundary,Objects):
    #provided a set of geometrical objects, say vertices or edges, this routine returns those that
    #are internal. 
    N=len(Objects)
    Internal=np.sort(np.array(list(set(np.arange(N))-set(Boundary))))  
    NumberInternal=len(Internal)
    return Internal,NumberInternal
    


#Assembly


        
def LocprojE(Func,Element,EdgeNodes,Nodes):
    #This function will, provided a function a set of nodes and edges, compute the 
    #projection onto the space of edge-based functions. The direction of the unit normal
    #will be assumed to be the clockwise rotation of the tangential vector.
    
    N=len(Element)
    proj=np.zeros((N,1))
    j = 0
    for i in Element:
        x1=Nodes[EdgeNodes[i][0]][0]
        y1=Nodes[EdgeNodes[i][0]][1]
        x2=Nodes[EdgeNodes[i][1]][0]
        y2=Nodes[EdgeNodes[i][1]][1]
        lengthe=math.sqrt((x2-x1)**2+(y2-y1)**2)
        xmid=0.5*(x1+x2)
        ymid=0.5*(y1+y2)
        etimesnormal=[y2-y1,x1-x2]
        Fx,Fy=Func(xmid,ymid)
        proj[j]=(etimesnormal[0]*Fx+etimesnormal[1]*Fy)*lengthe**-1  #midpoint rule
        j = j+1
    return proj



def LocalMassMatrix(N,R,n,A,nu):
    #Given the matrices N,R as defined in Ch.4 of MFD book and the dimension
    #of the reconstruction space this function assembles the local mass matrix
    #The formula is M=M0+M1 where M0=R(N^T R)^-1R^T and M1=lamb*DD^T where the 
    #columns of D span the null-space of N^T and lamb=2*trace(M0)/n 
    #n is the dimension of the reconstruction space
    #nu is the average, over the element, of the diffusion coefficient
    #A is the area of the element
    
    #These commands compute M0
    M0=np.matmul(np.transpose(N),R) 
    M0=np.linalg.inv(M0)
    M0=np.matmul(R,M0)
    M0=np.matmul(M0,np.transpose(R))
    
    #These commands compute M1
    #V=Matrix(np.transpose(N)).nullspace()
    #n=len(V)
    #k=len(M0[:,0])
    #D=np.zeros((k,n))
    #for i in range(n):
    #    for j in range(k):
    #        D[j,i]=V[i][j]
    #M1=np.matmul(D,np.transpose(D))
    #lamb=np.trace(M0)*2/n
    #M1=lamb*M1
    
    M1=np.linalg.inv(np.transpose(N).dot(N))
    M1=np.identity(n)-N.dot(M1).dot(np.transpose(N))
    
    gamma=np.trace(R.dot(np.transpose(R)))/(n*A*nu)
    #And finally we put the two matrices together
    return M0+M1*gamma
    
#The following three functions are necessary for the construction of the 
#mass matrix in the nodal space.
def m1(x,y,xP,yP):
        return 1

def m2(x,y,xP,yP):
        return x-xP

def m3(x,y,xP,yP):
        return y-yP

def P0(ElNodes,func,xP,yP):
        po = 0
        for node in ElNodes:
                x  = node[0]
                y  = node[1]
                po = po+func(x,y,xP,yP)
        return po/len(ElNodes)


def NewLocalMEWEMVWV(J,Basis,Element,EdgeNodes,Nodes,Ori):
    #This routine will compute the local mass matrix in the edge-based space E
    #Here we must ensure that the orientation of the elements is such that
    #We have an orientation for the edges that respects stoke's theorem
    n                      = len(Element)
    Dim                    = len(Basis)
    xP,yP,A,Vertices,Edges = Centroid(Element,EdgeNodes,Nodes,Ori)
    nu                     = DiffusionCoeff(xP,yP)
    NE                     = np.zeros((n,2))
    RE                     = np.zeros((n,2))
    
    for i in range(n):
        x1         = Vertices[i][0]
        y1         = Vertices[i][1]
        x2         = Vertices[i+1][0]
        y2         = Vertices[i+1][1]
        lengthEdge = math.sqrt((x2-x1)**2+(y2-y1)**2)
        NE[i][0]   = (y2-y1)*Ori[i]*lengthEdge**-1
        NE[i][1]   = (x1-x2)*Ori[i]*lengthEdge**-1
        RE[i][0]   = (0.5*(x1+x2)-xP)*Ori[i]*lengthEdge #These formulas are derived in the tex-document
        RE[i][1]   = (0.5*(y1+y2)-yP)*Ori[i]*lengthEdge
    ME = LocalMassMatrix(NE,RE,n,A,1)
    #WE=LocalMassMatrix(RE,NE,n,A,1)
    
    #########################
    #Here we will construct the local nodal mass matrix

    OrVert,OrEdg = StandardElement(Element,EdgeNodes,Nodes,Ori)
    ElNodes      = OrVert[0:n]    
    ElEdges      = OrEdg[0:n]
    G            = np.zeros((3,3))

    G[0,0] = 1

    G[0,1] = P0(ElNodes,m2,xP,yP)
    G[1,1] = A
    
    G[0,2] = P0(ElNodes,m3,xP,yP)
    G[2,2] = A

    B      = np.ones((3,n))/n
    H      = np.zeros((3,3))
    H[0,0] = A
            
    for i in range(n):
        x1         = Vertices[i][0]
        y1         = Vertices[i][1]
        x2         = Vertices[i+1][0]
        y2         = Vertices[i+1][1]
        
        lengthedge  = math.sqrt((x2-x1)**2+(y2-y1)**2)
        taux        = (x2-x1)/lengthedge
        tauy        = (y2-y1)/lengthedge

        B[1,i]      =  0.5*lengthedge*tauy
        B[2,i]      = -0.5*lengthedge*taux
        h           = lengthedge/3
        nx          = tauy
        ny          = -taux
        costheta    = (x2-x1)/lengthedge
        sintheta    = (y2-y1)/lengthedge
        
        xot         = x1+h*costheta
        yot         = y1+h*sintheta
        
        xtt         = x1+2*h*costheta
        ytt         = y1+2*h*sintheta
        
        H[1,1]      = H[1,1] + h*nx*( m2(x1,y1,xP,yP)**3+\
                                 3*m2(xot,yot,xP,yP)**3+\
                                 3*m2(xtt,ytt,xP,yP)**3+\
                                   m2(x2,y2,xP,yP)**3 )/8  

        H[2,2]      = H[2,2] + h*nx*( m3(x1,y1,xP,yP)**3+\
                                 3*m3(xot,yot,xP,yP)**3+\
                                 3*m3(xtt,ytt,xP,yP)**3+\
                                   m3(x2,y2,xP,yP)**3 )/8       
   

        H[1,2]      = H[1,2] + 3*h*nx*( m3(x1,y1,xP,yP)*m2(x1,y1,xP,yP)**2+\
                                   3*m3(xot,yot,xP,yP)*m2(xot,yot,xP,yP)**2+\
                                   3*m3(xtt,ytt,xP,yP)*m2(xtt,ytt,xP,yP)**2+\
                                     m3(x2,y2,xP,yP)*m2(x2,y2,xP,yP)**2 )/16

        H[2,1]      = H[2,1] + 3*h*nx*( m3(x1,y1,xP,yP)*m2(x1,y1,xP,yP)**2+\
                                   3*m3(xot,yot,xP,yP)*m2(xot,yot,xP,yP)**2+\
                                   3*m3(xtt,ytt,xP,yP)*m2(xtt,ytt,xP,yP)**2+\
                                     m3(x2,y2,xP,yP)*m2(x2,y2,xP,yP)**2 )/16

  
       
    D      = np.ones((n,3))

    D[:,1] = [m2(x,y,xP,yP) for [x,y] in ElNodes]    
    D[:,2] = [m3(x,y,xP,yP) for [x,y] in ElNodes]

    Pistar = np.linalg.inv(G).dot(B)
    Pi     = D.dot(Pistar)
    Id     = np.identity(n)
    MV     = np.transpose(Pistar).dot(H.dot(Pistar))+A*np.transpose(Id-Pi).dot(Id-Pi)    
   
    NJ = np.zeros((Dim,n))
    
    
    for i in range(Dim):        
        NJ[i,:] = np.transpose( LocprojE(Basis[i],Element,EdgeNodes,Nodes) )
    
    
    NJ = np.transpose(NJ)
    #print(NJ)
    b = np.transpose(NJ).dot(ME)
    #print(b)
    #print(ME)
    #print(NJ)
    #print(np.transpose(NJ).dot(ME).dot(NJ))
    #print(np.linalg.inv( np.transpose(NJ).dot(ME).dot(NJ) ) )
    
    MJ = np.linalg.pinv( np.transpose(NJ).dot(ME).dot(NJ) )
    #print(MJ)
    MJ = MJ.dot(b)
    #print(MJ)



    
    PolyCoordinates = np.zeros((2*(len(Vertices)-1),Dim))
    JMatrix = np.zeros( (len(Vertices)-1,2*(len(Vertices)-1)) ) 
                       
    l = 0
    k = 0
    for Polynomial in Basis:
        
        
        for j in range(len(Vertices)-1):
            Vertex = Vertices[j]
            x = Vertex[0]
            y = Vertex[1]
            
            Px,Py = Polynomial(x,y)
            
            PolyCoordinates[2*j,l] = Px
            PolyCoordinates[2*j+1,l] = Py
            
            
    
            if k==0:
                Jx,Jy = J(x,y)
                JMatrix[j,2*j] = -Jy
                JMatrix[j,2*j+1] = Jx
                
            j = j+1
        k = 1        
        l = l+1
    
    MJ = JMatrix.dot(PolyCoordinates).dot(MJ)
    MJ = MV.dot(MJ)
    return ME,MV,MJ,Edges

def NewAssembly(J,Basis,Nodes,EdgeNodes,ElementEdges,Orientations):
    #This routine takes a mesh and assembles the global mass matrices and their inverses
    
    
    NumberElements = len(ElementEdges)
    NumberEdges = len(EdgeNodes)
    NumberNodes = len(Nodes)
    
    
    ME = lil_matrix((NumberEdges,NumberEdges))
    #ME = np.zeros((NumberEdges,NumberEdges))
    #WE=np.zeros((NumberEdges,NumberEdges))
    MV = lil_matrix((NumberNodes,NumberNodes))
    #MV = np.zeros((NumberNodes,NumberNodes))
    #WV=np.zeros((NumberNodes,NumberNodes))
    MJ = lil_matrix((NumberNodes,NumberEdges))
    #MJ = np.zeros((NumberNodes,NumberEdges))
    
    #loop over the elements
    k = 0
    for Element in ElementEdges: 
        #Compute the local mass and stiffness matrices
        #LocME,LocWE,LocMV,LocWV,Edges=LocalMEWEMVWV(Element,EdgeNodes,Nodes) 
        Ori = Orientations[k]
        k = k+1
        LocME,LocMV,LocMJ,Edges = NewLocalMEWEMVWV(J,Basis,Element,EdgeNodes,Nodes,Ori) 
        
        #The assembly of the edge-based functions is easier since Element
        #Contains the order in which the edges ought to be assembled
       
        for j in range(len(Element)):
            ME[Element[j],Element] = ME[Element[j],Element] + LocME[j]
            #WE[Element[j],Element] = WE[Element[j],Element] + LocWE[j]
        n=len(Edges)-1
        ElementVertices = [0]*n
        
        for i in range(n):
            ElementVertices[i] = Edges[i][0]
         
        for j in range(len(Element)):
            MV[ElementVertices[j],ElementVertices] = MV[ElementVertices[j],ElementVertices]+LocMV[j]
            MJ[ElementVertices[j],Element] = MJ[ElementVertices[j],Element]+LocMJ[j]
            #WV[ElementVertices[j],ElementVertices]=WV[ElementVertices[j],ElementVertices]+LocWV[j]
        
        #print(MJ)
        #i = 0
        #print(np.linalg.norm(LocMJ))
        #for Edge in Element:
        #    MJ[Edge,ElementVertices] = MJ[Edge,ElementVertices]+LocMJ[i]
        #    i = i+1
        #for j in range(len(Element)):
            #MJ[ElementVertices[j],Element] = MJ[ElementVertices[j],Element]+LocMJ[j]
    #return ME,WE,MV,WV
    
    ME = ME.tocsr()
    MV = MV.tocsr()
    MJ = MJ.tocsr()
    
    return ME,MV,MJ

#Interpolators 

def projV(func,Nodes):
    #This function will, provided a function and a set of nodes, compute the projection onto 
    #the space of node-based functions.
    n    = len(Nodes)
    proj = np.zeros((n,1))
    for i in range(n):
        proj[i] = func(Nodes[i][0],Nodes[i][1])
    return proj

def projE(Func,EdgeNodes,Nodes):
    #This function will, provided a function a set of nodes and edges, compute the 
    #projection onto the space of edge-based functions. The direction of the unit normal
    #will be assumed to be the clockwise rotation of the tangential vector.
    
    N=len(EdgeNodes)
    proj=np.zeros((N,1))
    for i in range(N):
        x1=Nodes[EdgeNodes[i][0]][0]
        y1=Nodes[EdgeNodes[i][0]][1]
        x2=Nodes[EdgeNodes[i][1]][0]
        y2=Nodes[EdgeNodes[i][1]][1]
        lengthe=math.sqrt((x2-x1)**2+(y2-y1)**2)
        xmid=0.5*(x1+x2)
        ymid=0.5*(y1+y2)
        etimesnormal=[y2-y1,x1-x2]
        Fx,Fy=Func(xmid,ymid)
        proj[i]=(etimesnormal[0]*Fx+etimesnormal[1]*Fy)*lengthe**-1  #midpoint rule
    return proj


#PrimaryOperators

def primcurl(EdgeNodes,Nodes):
    #This routine computes the primary curl as a matrix
    nN = len(Nodes)
    nE = len(EdgeNodes)
    #curl = np.zeros((nE,nN))
    curl = lil_matrix((nE,nN))
    for i in range(nE):
        Node1 = EdgeNodes[i][0]
        Node2 = EdgeNodes[i][1]
        x1 = Nodes[Node1][0]
        y1 = Nodes[Node1][1]
        x2 = Nodes[Node2][0]
        y2 = Nodes[Node2][1]
        lengthe = math.sqrt((x2-x1)**2+(y2-y1)**2)
        
        curl[i,Node2] = 1/lengthe #These formulas are derived in the pdf document
        curl[i,Node1] = -1/lengthe
    curl = curl.tocsr()
    return curl

def primdiv(ElementEdges,EdgeNodes,Nodes):
    #this routine computes the primary divergence matrix
    NEl=len(ElementEdges)
    NE=len(EdgeNodes)
    div=np.zeros((NEl,NE))
    for i in range(NEl):
        Element=ElementEdges[i]
        N=len(Element)
        for j in range(N):
            Node1=EdgeNodes[Element[j]][0]
            Node2=EdgeNodes[Element[j]][1]
            x1=Nodes[Node1][0]
            y1=Nodes[Node1][1] #these formulas are derived in the pdf document
            x2=Nodes[Node2][0]
            y2=Nodes[Node2][1]
            lengthe=math.sqrt((x2-x1)**2+(y2-y1)**2)
            xP,yP,A,Vertices,Edges,Ori=Centroid(Element,EdgeNodes,Nodes)
            div[i,Element[j]]=Ori[j]*lengthe/A
    return div


#Solver


def NewSolver(J,Basis,Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt,theta):
    #This routine will, provided a mesh, final time and time step, return the values of the electric and magnetic field at
    #the given time.
    #The boundary conditions are given above 
    time = np.arange(0,T,dt)
    InternalNodes,NumberInternalNodes = InternalObjects(BoundaryNodes,Nodes)
    ME,MV,MJ = NewAssembly(J,Basis,Nodes,EdgeNodes,ElementEdges,Orientations) #compute the mass matrices
    
    
    #Let us construct the required matrices
    
    curl = primcurl(EdgeNodes,Nodes) #the primary curl
    #D = np.zeros((len(Nodes),len(Nodes))) #this matrix will is explained in the pdf
    D = lil_matrix((len(Nodes),len(Nodes)))
    for i in InternalNodes:
        D[i,i]=1
    D = D.tocsr()
   
    Aprime = MV+theta*dt*( ( np.transpose(curl) ).dot(ME)+MJ ).dot(curl)#MV.dot(MJ) ).dot(curl)
    Aprime = D.dot(Aprime)
    #A = np.zeros((NumberInternalNodes,NumberInternalNodes))
    A = lil_matrix((NumberInternalNodes,NumberInternalNodes))
    

    for i in range(NumberInternalNodes):
        A[i,:] = Aprime[InternalNodes[i],InternalNodes]
    A = A.tocsr()
    
    b = np.transpose(curl).dot(ME)+MJ#+MV.dot(MJ)
    b = D.dot(b)
    
    Bh = projE(InitialCond,EdgeNodes,Nodes)
    Bh = np.transpose(Bh)[0]
  
    Eh = np.zeros(len(Nodes))
    
    
    EhInterior = np.zeros(len(Nodes)) #This is Eh in the interior
    
    EhBoundary = np.zeros(len(Nodes))   #This is Eh on the boundary
   
    
    
    for t in time:
        
        #We update the time dependant boundary conditions
        #i.e. The boundary values of the electric field
        for NodeNumber in BoundaryNodes:
            Node = Nodes[NodeNumber]
            EhBoundary[NodeNumber] = EssentialBoundaryCond(Node[0],Node[1],t+0.5*dt)
        
        #Solve  for the internal values of the electric field
        
        W1 = b.dot(Bh)
        W2 = Aprime.dot(EhBoundary)
        
        #EhInterior[InternalNodes] = np.linalg.solve(A,W1[InternalNodes]-W2[InternalNodes]) 
        
        #EhInterior[InternalNodes] = spsolve(A,W1[InternalNodes]-W2[InternalNodes]) 
        #f = spsolve(A,W1[InternalNodes]-W2[InternalNodes]) 
        #EhInterior[InternalNodes] = f
        
        EhInterior[InternalNodes] = spsolve(A,W1[InternalNodes]-W2[InternalNodes]) 
        
        Eh = EhInterior+EhBoundary
        
        
        #Update the magnetic field
        Bh = Bh-dt*curl.dot(Eh) 
           
        
    #Now we compute the error
    def ContB(x,y):
        return ExactB(x,y,T)
    def ContE(x,y):
        return ExactE(x,y,T-0.5*dt)
    
    Bex = projE(ContB,EdgeNodes,Nodes)
    Eex = projV(ContE,Nodes)
    
    Bex = np.transpose(Bex)[0]
    Eex = np.transpose(Eex)[0]
    
    B = Bh-Bex
    E = Eh-Eex
    #MagneticError = np.transpose(B).dot(ME).dot(B)
    #ElectricError = np.transpose(E).dot(MV).dot(E)
    
    MagneticError = ME.dot(B).dot(B)
    ElectricError = MV.dot(E).dot(E)
    
    #MagneticError = math.sqrt(MagneticError[0,0])
    #ElectricError = math.sqrt(ElectricError[0,0])
    
    MagneticError = math.sqrt(MagneticError)
    ElectricError = math.sqrt(ElectricError)
    
    return Bh,Eh,MagneticError,ElectricError


