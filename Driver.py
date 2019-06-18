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
from Functions import *






ProcessedFiles = ['PVh=0.128037.txt','PVh=0.0677285.txt','PVh=0.124524.txt','PVh=0.221367.txt',\
                  'PVh=0.0633169.txt']#,'PVh=0.0314634.txt','PVh=0.0165378.txt']

h = [0.12803687993289598, 0.06772854614785964, 0.03450327796711771, 0.017476749542968805,\
     0.008787156237382746]#, 0.004419676355414694,0.0022139558199118672]




Basis = [Poly1,Poly2,Poly]
T=0.25
#h=[1/(2**(2+i)) for i in range(len(Files))]
FiveVoronoiElectricError=[0]*len(ProcessedFiles)
FiveVoronoiMagneticError=[0]*len(ProcessedFiles)
i=0
for Pfile in ProcessedFiles:
    dt=0.05*h[i]**2
    #dt = h[i]**2
    #dt = h[i]
    print(Pfile)
    Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations=ProcessedMesh(Pfile)

    Bh,Eh,Berror,Eerror = NewSolver(J,Basis,Nodes,EdgeNodes,ElementEdges,\
                                    BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt)
    #Bh,Eh,Berror,Eerror=Solver(Nodes,EdgeNodes,ElementEdges,BoundaryNodes,\
                               #EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt)
    print('Computed Numerical Solution')
    #VisualizeE(Eh,Nodes)
    FiveVoronoiElectricError[i]=Eerror
    FiveVoronoiMagneticError[i]=Berror
    i=i+1

with open('FiveEMVoronoi.txt', "wb") as fp:   #Pickling
    pickle.dump([FiveVoronoiElectricError,FiveVoronoiMagneticError], fp)




#Triangles

ProcessedFiles = ['PTh=0.0179733.txt']#,'PTh=0.0089405.txt']


h = [0.0031355820733239914]#,0.0015683308166871686]

Basis = [Poly1,Poly2,Poly]
T=0.25
#h=[1/(2**(2+i)) for i in range(len(Files))]
twoTriangleElectricError=[0]*len(ProcessedFiles)
twoTriangleMagneticError=[0]*len(ProcessedFiles)
i=0


for Pfile in ProcessedFiles:
    dt=0.05*h[i]**2
    #dt = h[i]
    print(Pfile)
    Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations=ProcessedMesh(Pfile)
    #print('Retrieved The Mesh')
    Bh,Eh,Berror,Eerror = NewSolver(J,Basis,Nodes,EdgeNodes,ElementEdges,\
                                    BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt)
    #Bh,Eh,Berror,Eerror=Solver(Nodes,EdgeNodes,ElementEdges,BoundaryNodes,\
                               #EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt)
    print('Computed Numerical Solution')
    #VisualizeE(Eh,Nodes)
    twoTriangleElectricError[i]=Eerror
    twoTriangleMagneticError[i]=Berror
    i=i+1


with open('twoTriangleEMError.txt', "wb") as fp:   #Pickling
    pickle.dump([twoTriangleElectricError,twoTriangleMagneticError], fp)



#Squares



ProcessedFiles = ['PertPQh=0.0155408.txt']#,'PertPQh=0.00779181.txt']


h = [0.005494505494505494]#,0.0027548209366391185]

Basis = [Poly1,Poly2,Poly]
T=0.25

twoQuadElectricError=[0]*len(ProcessedFiles)
twoQuadMagneticError=[0]*len(ProcessedFiles)
i=0
for Pfile in ProcessedFiles:
    dt=0.05*h[i]**2
    #dt = h[i]**2
    #dt = h[i]
    print(Pfile)
    Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations=ProcessedMesh(Pfile)

    Bh,Eh,Berror,Eerror = NewSolver(J,Basis,Nodes,EdgeNodes,ElementEdges,\
                                    BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt)
    #Bh,Eh,Berror,Eerror=Solver(Nodes,EdgeNodes,ElementEdges,BoundaryNodes,\
                               #EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt)
    print('Computed Numerical Solution')
    VisualizeE(Eh,Nodes)
    twoQuadElectricError[i]=Eerror
    twoQuadMagneticError[i]=Berror
    i=i+1


with open('twoQuadEMError.txt', "wb") as fp:   #Pickling
    pickle.dump([twoQuadElectricError,twoQuadMagneticError], fp)



#Voronoi



ProcessedFiles = ['PVh=0.0314634.txt']#,'PVh=0.0165378.txt']

h = [0.004419676355414694]#,0.0022139558199118672]




Basis = [Poly1,Poly2,Poly]
T=0.25
#h=[1/(2**(2+i)) for i in range(len(Files))]
twoVoronoiElectricError=[0]*len(ProcessedFiles)
twoVoronoiMagneticError=[0]*len(ProcessedFiles)
i=0
for Pfile in ProcessedFiles:
    dt=0.05*h[i]**2
    #dt = h[i]**2
    #dt = h[i]
    print(Pfile)
    Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations=ProcessedMesh(Pfile)

    Bh,Eh,Berror,Eerror = NewSolver(J,Basis,Nodes,EdgeNodes,ElementEdges,\
                                    BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt)
    #Bh,Eh,Berror,Eerror=Solver(Nodes,EdgeNodes,ElementEdges,BoundaryNodes,\
                               #EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt)
    print('Computed Numerical Solution')
    #VisualizeE(Eh,Nodes)
    twoVoronoiElectricError[i]=Eerror
    twoVoronoiMagneticError[i]=Berror
    i=i+1

with open('twoEMVoronoi.txt', "wb") as fp:   #Pickling
    pickle.dump([twoVoronoiElectricError,twoVoronoiMagneticError], fp)

