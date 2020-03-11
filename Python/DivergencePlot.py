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

theta          = 0.5
ProcessedFiles = ['PTh=0.051886.txt','PertPQh=0.043478.txt','PVh=0.0677285.txt']
h = [0.051886,0.043478,0.0677285]
Basis = [Poly1,Poly2,Poly]
T     = 20
y     = 0
for Pfile in ProcessedFiles:
    print(Pfile)
    Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations = ProcessedMesh(Pfile)
    dt     = 0.05*h[y]**2

    DivMat = primdiv(ElementEdges,EdgeNodes,Nodes,Orientations)

    time   = np.arange(0,T,dt)
    InternalNodes,NumberInternalNodes = InternalObjects(BoundaryNodes,Nodes)
    ME,MV,MJ = MFDAssembly(J,Nodes,EdgeNodes,ElementEdges,Orientations) #compute the mass matrices
    #ME,MV,MJ = NewAssembly(J,Basis,Nodes,EdgeNodes,ElementEdges,Orientations) #compute the mass matrices
    #ME,MV,MJ = LeastSquaresAssembly(J,Basis,Nodes,EdgeNodes,ElementEdges,Orientations)
    #ME,MV,MJ = PiecewiseAssembly(J,Basis,Nodes,EdgeNodes,ElementEdges,Orientations)
    
    #print('Piecewise Matrices Assembled')

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
    Bh = HighOrder7projE(InitialCond,EdgeNodes,Nodes)
    Bh = np.transpose(Bh)[0]
    Eh = np.zeros(len(Nodes))   
    EhInterior = np.zeros(len(Nodes)) #This is Eh in the interior  
    EhBoundary = np.zeros(len(Nodes))   #This is Eh on the boundary  
    Divergence = time*0    
    l          = 0
    NEl        = len(ElementEdges)
    print('In time integration')
    for t in time:
    #Here we compute the square of the L^2 error
        divB                         = DivMat.dot(Bh)
        DivErr                       = 0
    
        for j in range(NEl):
    
            Element                = ElementEdges[j]
            Ori                    = Orientations[j]
            xP,yP,Ar,Vertices,Edges = Centroid(Element,EdgeNodes,Nodes,Ori)
            DivErr                 = DivErr + Ar*(divB[j]**2)
        Divergence[l] = DivErr
        l             = l+1   
        #We update the time dependant boundary conditions
        #i.e. The boundary values of the electric field
        for NodeNumber in BoundaryNodes:
            Node = Nodes[NodeNumber]
            EhBoundary[NodeNumber] = EssentialBoundaryCond(Node[0],Node[1],t+theta*dt)
        W1 = b.dot(Bh)
        W2 = Aprime.dot(EhBoundary)  
        EhInterior[InternalNodes] = spsolve(A,W1[InternalNodes]-W2[InternalNodes])    
        Eh = EhInterior+EhBoundary
        Bh = Bh-dt*curl.dot(Eh)
     
    # divB                         = DivMat.dot(Bh)
    # DivErr                       = 0
    
    # for j in range(NEl):
    
    #     Element                = ElementEdges[j]
    #     Ori                    = Orientations[j]
    #     xP,yP,Ar,Vertices,Edges = Centroid(Element,EdgeNodes,Nodes,Ori)
    #     DivErr                 = DivErr + Ar*(divB[j]**2)
    # Divergence[l] = DivErr
    #Now we record the time vector and divergence vextor in a text file.
    print('recording results')
    if y == 0:
        with open('MFDTrigSimulations.m','w') as file:
        #saving time first
            file.writelines('trigtime = [')
            for e in range(len(time)):
                if e == 0:
                    file.writelines(str(time[e]))
                else:
                    file.writelines(',')
                    file.writelines(str(time[e]))
        
            file.writelines('];')
            file.write('\n')

            file.writelines('trigdiv = [')
            for e in range(len(Divergence)):
                if e == 0:
                    file.writelines(str(Divergence[e]))
                else:
                    file.writelines(',')
                    file.writelines(str(Divergence[e]))
        
            file.writelines('];')
            file.write('\n')

    if y == 1:
        with open('MFDQuadSimulations.m','w') as file:
        #saving time first
            file.writelines('quadtime = [')
            for e in range(len(time)):
                if e == 0:
                    file.writelines(str(time[e]))
                else:
                    file.writelines(',')
                    file.writelines(str(time[e]))
        
            file.writelines('];')
            file.write('\n')

            file.writelines('quaddiv = [')
            for e in range(len(Divergence)):
                if e == 0:
                    file.writelines(str(Divergence[e]))
                else:
                    file.writelines(',')
                    file.writelines(str(Divergence[e]))
        
            file.writelines('];')
            file.write('\n')
        
    if y == 2:
        with open('MFDVoronoiSimulations.m','w') as file:
        #saving time first
            file.writelines('voronoitime = [')
            for e in range(len(time)):
                if e == 0:
                    file.writelines(str(time[e]))
                else:
                    file.writelines(',')
                    file.writelines(str(time[e]))
        
            file.writelines('];')
            file.write('\n')

            file.writelines('voronoidiv = [')
            for e in range(len(Divergence)):
                if e == 0:
                    file.writelines(str(Divergence[e]))
                else:
                    file.writelines(',')
                    file.writelines(str(Divergence[e]))
        
            file.writelines('];')
            file.write('\n')
    y      = y+1
    print('Finished')