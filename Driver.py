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


##############################################################Voronoi

theta = 0.5

ProcessedFiles = ['PVh=0.128037.txt','PVh=0.0677285.txt','PVh=0.0345033.txt','PVh=0.0174767.txt',\
                  'PVh=0.0087872.txt']

h = [0.12803687993289598, 0.06772854614785964, 0.03450327796711771, 0.017476749542968805,\
     0.008787156237382746]




Basis                    = [Poly1,Poly2,Poly]
T                        = 0.25

FiveVoronoiElectricError = [0]*len(ProcessedFiles)
FiveVoronoiMagneticError = [0]*len(ProcessedFiles)
i                        = 0
for Pfile in ProcessedFiles:
    dt = 0.05*h[i]**2
    #dt = h[i]**2
    #dt = h[i]
    print(Pfile)
    Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations = ProcessedMesh(Pfile)

    Bh,Eh,Berror,Eerror = NewSolver(J,Basis,Nodes,EdgeNodes,ElementEdges,\
                                    BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt,theta)
    #Bh,Eh,Berror,Eerror=Solver(Nodes,EdgeNodes,ElementEdges,BoundaryNodes,\
                               #EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt)
    print('Computed Numerical Solution')
    #VisualizeE(Eh,Nodes)
    FiveVoronoiElectricError[i] = Eerror
    FiveVoronoiMagneticError[i] = Berror
    i=i+1

#with open('FiveEMVoronoi.txt', "wb") as fp:   #Pickling
#    pickle.dump([FiveVoronoiElectricError,FiveVoronoiMagneticError], fp)

print('Voronoi Electric = '+str(FiveVoronoiElectricError))
print('Voronoi Magnetic = '+str(FiveVoronoiMagneticError))



#########################################Triangles

ProcessedFiles = ['PTh=0.101015.txt','PTh=0.051886.txt','PTh=0.0251418.txt','PTh=0.0125255.txt',\
         'PTh=0.0062613.txt']
 
 
h = [0.10101525445522107, 0.05018856132284956, 0.025141822757713456, 0.012525468249897755,\
     0.006261260829309998]

Basis = [Poly1,Poly2,Poly]
T     = 0.25
#h=[1/(2**(2+i)) for i in range(len(Files))]
FiveTriangleElectricError = [0]*len(ProcessedFiles)
FiveTriangleMagneticError = [0]*len(ProcessedFiles)
i=0


for Pfile in ProcessedFiles:
    dt=0.05*h[i]**2
    #dt = h[i]
    print(Pfile)
    Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations=ProcessedMesh(Pfile)
    #print('Retrieved The Mesh')
    Bh,Eh,Berror,Eerror = NewSolver(J,Basis,Nodes,EdgeNodes,ElementEdges,\
                                    BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt,theta)
    #Bh,Eh,Berror,Eerror=Solver(Nodes,EdgeNodes,ElementEdges,BoundaryNodes,\
                               #EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt)
    print('Computed Numerical Solution')
    #VisualizeE(Eh,Nodes)
    FiveTriangleElectricError[i]=Eerror
    FiveTriangleMagneticError[i]=Berror
    i=i+1


print('Triangle Electric = '+str(FiveTriangleElectricError))
print('Triangle Magnetic = '+str(FiveTriangleMagneticError))



#######################Quadrilaterals

     
ProcessedFiles = ['PertPQh=0.166666.txt','PertPQh=0.0833333.txt','PertPQh=0.043478.txt',\
                  'PertPQh=0.021739.txt','PertPQh=0.010989.txt']
 
h = [0.16666666666666666, 0.08333333333333333, 0.043478260869565216, 0.021739130434782608,\
     0.010989010989010988]


Basis = [Poly1,Poly2,Poly]
T     = 0.25

FiveQuadElectricError=[0]*len(ProcessedFiles)
FiveQuadMagneticError=[0]*len(ProcessedFiles)
i=0
for Pfile in ProcessedFiles:
    dt=0.05*h[i]**2
    #dt = h[i]**2
    #dt = h[i]
    print(Pfile)
    Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations=ProcessedMesh(Pfile)

    Bh,Eh,Berror,Eerror = NewSolver(J,Basis,Nodes,EdgeNodes,ElementEdges,\
                                    BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt,theta)
    #Bh,Eh,Berror,Eerror=Solver(Nodes,EdgeNodes,ElementEdges,BoundaryNodes,\
                               #EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt)
    print('Computed Numerical Solution')
    FiveQuadElectricError[i]=Eerror
    FiveQuadMagneticError[i]=Berror
    i=i+1

print('Quad Electric = '+str(FiveQuadElectricError))
print('Quad Magnetic = '+str(FiveQuadMagneticError))

