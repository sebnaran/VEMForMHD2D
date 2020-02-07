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

#Tasks = ['PW','H1','LS']
#Theta = [0.5,1]
Tasks  = ['H1']
Thetas = [0.5]
#Final time
T     = 0.5
#Test Q's
Q     = range(1,10,0.1)
#ProcessedFiles = ['PTh=0.051886.txt','PertPQh=0.043478.txt','PVh=0.0677285.txt']
ProcessedFiles = ['PTh=0.051886.txt']
#h = [0.051886,0.043478,0.0677285]
h = [0.0677285]

for task in Tasks:
    print(task)
    FileCounter = 0
    for Pfile in ProcessedFiles:
        print(Pfile)
        dt           = h[FileCounter]
        FileCounter= = FileCounter+1
        time  = np.arange(0,T,dt)
        
        Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations = ProcessedMesh(Pfile)  
        

        if task == 'H1':
            ME,MV,MJ = EAssembly(J,Nodes,EdgeNodes,ElementEdges,Orientations) #compute the mass matrices
        elif task == 'LS':
            ME,MV,MJ = LSAssembly(J,Nodes,EdgeNodes,ElementEdges,Orientations)
        elif task == 'GI':
            ME,MV,MJ = GIAssembly(J,Basis,Nodes,EdgeNodes,ElementEdges,Orientations)

        #We prepare the necessary matrices to set up the linear systen.
        # curl = primcurl(EdgeNodes,Nodes)
        # D    = lil_matrix((len(Nodes),len(Nodes)))
        # for i in InternalNodes:
        #     D[i,i]=1
        # D = D.tocsr()

        # Aprime = MV+theta*dt*( ( np.transpose(curl) ).dot(ME)+MJ ).dot(curl)#MV.dot(MJ) ).dot(curl)
        # Aprime = D.dot(Aprime)
        # A      = lil_matrix((NumberInternalNodes,NumberInternalNodes))
        
        # for i in range(NumberInternalNodes):
        #     A[i,:] = Aprime[InternalNodes[i],InternalNodes]
        # A = A.tocsr()
    
        # b = np.transpose(curl).dot(ME)+MJ#+MV.dot(MJ)
        # b = D.dot(b)

        # Bh   = HighOrder7projE(InitialCond,EdgeNodes,Nodes)
        # Bh   = np.transpose(Bh)[0]
        # RHS1 = Bh.dot(ME.dot(Bh))

        # Eh         = np.zeros(len(Nodes))
        # EhInterior = np.zeros(len(Nodes))
        # EhBoundary = np.zeros(len(Nodes))
        # step       = 1
        # for t in time[0:len(time)-1]:

        #     for NodeNumber in BoundaryNodes:
        #         Node = Nodes[NodeNumber]
        #         EhBoundary[NodeNumber] = EssentialBoundaryCond(Node[0],Node[1],t+theta*dt)

        #     RHS2 = RHS2+\
        #            (beta**step)*gamma*dt*\
        #            ( EhBoundary.dot(MV.dot(EhBoundary))+\
        #            (curl.dot(EhBoundary)).dot(ME.dot(curl.dot(EhBoundary))) )

        #     W1 = b.dot(Bh)
        #     W2 = Aprime.dot(EhBoundary)

        #     EhInterior[InternalNodes] = spsolve(A,W1[InternalNodes]-W2[InternalNodes])

        #     Eh   = EhInterior+EhBoundary
        #     LHS2 = LHS2+0.5*gamma*dt(beta**(step))*Eh.dot(MV.dot(Eh))

        #     Bh   = Bh-dt*curl.dot(Eh)
        #     LHS1 = (beta**step)*Bh.dot(ME.dot(Bh))

        #     LHS[step] = LHS1+LHS2
        #     RHS[step] = RHS1+RHS2
        #     step      = step+1

#         print('recording results')
#         if y == 0:
#         with open('TrigSimulations.m','w') as file:
#         #saving time first
#             file.writelines('trigtime = [')
#             for e in range(len(time)):
#                 if e == 0:
#                     file.writelines(str(time[e]))
#                 else:
#                     file.writelines(',')
#                     file.writelines(str(time[e]))
        
#             file.writelines('];')
#             file.write('\n')

#             file.writelines('trigdiv = [')
#             for e in range(len(Divergence)):
#                 if e == 0:
#                     file.writelines(str(Divergence[e]))
#                 else:
#                     file.writelines(',')
#                     file.writelines(str(Divergence[e]))
        
#             file.writelines('];')
#             file.write('\n')

#     if y == 1:
#         with open('QuadSimulations.m','w') as file:
#         #saving time first
#             file.writelines('quadtime = [')
#             for e in range(len(time)):
#                 if e == 0:
#                     file.writelines(str(time[e]))
#                 else:
#                     file.writelines(',')
#                     file.writelines(str(time[e]))
        
#             file.writelines('];')
#             file.write('\n')

#             file.writelines('quaddiv = [')
#             for e in range(len(Divergence)):
#                 if e == 0:
#                     file.writelines(str(Divergence[e]))
#                 else:
#                     file.writelines(',')
#                     file.writelines(str(Divergence[e]))
        
#             file.writelines('];')
#             file.write('\n')
        
#     if y == 2:
#         with open('VoronoiSimulations.m','w') as file:
#         #saving time first
#             file.writelines('voronoitime = [')
#             for e in range(len(time)):
#                 if e == 0:
#                     file.writelines(str(time[e]))
#                 else:
#                     file.writelines(',')
#                     file.writelines(str(time[e]))
        
#             file.writelines('];')
#             file.write('\n')

#             file.writelines('voronoidiv = [')
#             for e in range(len(Divergence)):
#                 if e == 0:
#                     file.writelines(str(Divergence[e]))
#                 else:
#                     file.writelines(',')
#                     file.writelines(str(Divergence[e]))
        
#             file.writelines('];')
#             file.write('\n')
#     y      = y+1
#     print('Finished')   