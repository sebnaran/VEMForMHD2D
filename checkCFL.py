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

theta = 0.25

#ProcessedFiles = ['PVh=0.128037.txt','PVh=0.0677285.txt','PVh=0.0345033.txt','PVh=0.0174767.txt',\
#                  'PVh=0.0087872.txt']

#h = [0.12803687993289598, 0.06772854614785964, 0.03450327796711771, 0.017476749542968805,\
#     0.008787156237382746]
ProcessedFiles = ['PTh=0.101015.txt','PTh=0.051886.txt','PTh=0.0251418.txt','PTh=0.0125255.txt',\
         'PTh=0.0062613.txt']


h = [0.10101525445522107, 0.05018856132284956, 0.025141822757713456, 0.012525468249897755,\
     0.006261260829309998]


Basis = [Poly1,Poly2,Poly]
T     = 0.05
CFL   = 150
i     = 1 
if True:
#for CFL in range(500,5,-1):
#    CFL   = 0.0001*CFL
    dt    = CFL*h[i]**2
    Pfile = ProcessedFiles[i]
    Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations = ProcessedMesh(Pfile)

    Bh,Eh,Berror,Eerror = NewSolver(J,Basis,Nodes,EdgeNodes,ElementEdges,\
                                    BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt,theta)
    if Berror+Eerror < 100:
        print(Berror)
        print(Eerror)
        print(CFL)
#        break
