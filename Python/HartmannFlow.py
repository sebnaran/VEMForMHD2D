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
from EnergyClass import Energy
from Functions import *

T     = 10
#Pfile = 'HartPVh=0.0347734.txt'
#Pfile = 'HartPVh=0.0124216.txt'
Pfile = 'PTh=0.0251418.txt'
#Pfile = 'PVh=0.0174767.txt' 
theta = 0.5
dt    = 0.005
task  = 'E'
Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations = ProcessedMesh(Pfile)

#Bh,Eh,Berror,Eerror = ESolver(J,Nodes,EdgeNodes,ElementEdges,\
#                                        BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt,theta)
#Bh,Berror = HartSolver(J,Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt,theta)
#q  = 0
#Ex = [0]*len(Nodes)
#Ey = [0]*len(Nodes)

#for Node in Nodes:
#    Ex[q] = Node[0]
#    Ey[q] = Node[1]
#    q     = q+1

#SaveInmFile('HEcoorx','Ecoorx',Ex)
#SaveInmFile('HEcoory','Ecoory',Ey)
#SaveInmFile('HE','E',Eh)

Bx = [0]*len(ElementEdges)
By = [0]*len(ElementEdges)
x  = [0]*len(ElementEdges)
y  = [0]*len(ElementEdges)
w  = 0

for Element in ElementEdges:
    Basis          = [Poly1,Poly2,Poly]
    Ori = Orientations[w]
    ME,MV,MJ,Edges = NewLocalMEWEMVWV(J,Basis,Element,EdgeNodes,Nodes,Ori)
    
    n   = len(Element)
    Dim = len(Basis)
    NJ  = np.zeros((Dim,n))
    for i in range(Dim):        
        NJ[i,:] = np.transpose( LocprojE(Basis[i],Element,EdgeNodes,Nodes) )

    NJ = np.transpose(NJ)
    A  = np.transpose(NJ).dot(ME).dot(NJ)

    b  = np.transpose(NJ).dot(ME).dot(Bh[Element])

    C  = np.linalg.inv(A).dot(b)
    
    xP,yP,A,Vertices,Edges = Centroid(Element,EdgeNodes,Nodes,Ori)
    x[w]  = xP
    y[w]  = yP
    Bx[w] = C[0]+C[2]*xP
    By[w] = C[1]+C[2]*yP
    w     = w+1

#SaveInmFile('HBcoorx','Bcoorx',x)
#SaveInmFile('HBcoory','Bcoory',y)
#SaveInmFile('HBx','Bx',Bx)
#SaveInmFile('HBy','By',By)

print(Berror)
#print(Eerror)