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


Tasks = ['GI','E','LS']
for task in Tasks:
    print(task)
    theta = 0.5
    ProcessedFiles = ['PVh=0.128037.txt','PVh=0.0677285.txt','PVh=0.0345033.txt','PVh=0.0174767.txt',\
                      'PVh=0.0087872.txt']

    h = [0.12803687993289598, 0.06772854614785964, 0.03450327796711771, 0.017476749542968805,\
        0.008787156237382746]
    #ProcessedFiles = ['PVh=0.00442309.txt','PVh=0.00221383.txt']
    #h              = [0.00442309,0.00221383]
    Basis = [Poly1,Poly2,Poly]
    T     = 0.25
    VoronoiElectricErr = [0]*len(ProcessedFiles)
    VoronoiMagneticErr = [0]*len(ProcessedFiles)
    #DivTriangleErr      = [0]*len(ProcessedFiles)
    i                   = 0
    for Pfile in ProcessedFiles:
        dt = 0.05*h[i]**2
        #dt = h[i]
        print(Pfile)
        Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations = ProcessedMesh(Pfile)
        #Nodes,EdgeNodes,ElementEdges,Orientations = ProcessedMeshNB(Pfile)
        #DivMat                                                  = primdiv(ElementEdges,EdgeNodes,Nodes,Orientations)
        #print('Retrieved The Mesh')
        if task == 'E':
            Bh,Eh,Berror,Eerror = ESolver(J,Nodes,EdgeNodes,ElementEdges,\
                                        BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt,theta)
        if task == 'LS':
            Bh,Eh,Berror,Eerror = LSSolver(J,Nodes,EdgeNodes,ElementEdges,\
            BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,\
                ExactE,ExactB,T,dt,theta)
        if task == 'GI':
            Bh,Eh,Berror,Eerror = GISolver(J,Nodes,EdgeNodes,ElementEdges,\
            BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,\
                ExactE,ExactB,T,dt,theta)        
        #VisualizeE(Eh,Nodes)
        VoronoiElectricErr[i] = Eerror
        VoronoiMagneticErr[i] = Berror
        
        #    divB                         = DivMat.dot(Bh)
        #    DivErr                       = 0
        #    NEl                          = len(ElementEdges)
        #    for j in range(NEl):
        #        Element                = ElementEdges[j]
        #        Ori                    = Orientations[j]
        #        xP,yP,A,Vertices,Edges = Centroid(Element,EdgeNodes,Nodes,Ori)
        #        DivErr                 = DivErr + A*(divB[j]**2)
        #    #DivErr                    = math.sqrt(DivErr)
        #    DivTriangleErr[i] = DivErr
        i = i+1
        #with open('Vornoi'+task+'.txt', "wb") as fp:
        #    pickle.dump((VoronoiElectricErr,VoronoiMagneticErr),fp)

    print('Voronoi Electric = '+str(VoronoiElectricErr))
    print('Voronoi Magnetic = '+str(VoronoiMagneticErr))
    #print('Voronoi Divergence err ='+str(DivTriangleErr))
    ####################################################################################################
    ####################################################################################################
    ProcessedFiles = ['PertPQh=0.166666.txt','PertPQh=0.0833333.txt','PertPQh=0.043478.txt',\
                   'PertPQh=0.021739.txt','PertPQh=0.010989.txt']

    h = [0.16666666666666666, 0.08333333333333333, 0.043478260869565216, 0.021739130434782608,\
       0.010989010989010988]
    # ProcessedFiles = ['PertPQh=0.00549451.txt','PertQh=0.00275482.txt']
    # h              = [0.00549451,0.00275482]
    Basis = [Poly1,Poly2,Poly]
    T     = 0.25

    QuadElectricErr = [0]*len(ProcessedFiles)
    QuadMagneticErr = [0]*len(ProcessedFiles)
    #DivQuadErr      = [0]*len(ProcessedFiles)
    i = 0
    for Pfile in ProcessedFiles:
        dt = 0.05*h[i]**2
        #dt = h[i]**2
        #dt = h[i]
        print(Pfile)
        Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations = ProcessedMesh(Pfile)
        #Nodes,EdgeNodes,ElementEdges,Orientations = ProcessedMeshNB(Pfile)
        #DivMat                                                  = primdiv(ElementEdges,EdgeNodes,Nodes,Orientations)
        if task == 'E':
            Bh,Eh,Berror,Eerror = ESolver(J,Nodes,EdgeNodes,ElementEdges,\
                                        BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt,theta)
        if task == 'LS':
            Bh,Eh,Berror,Eerror = LSSolver(J,Nodes,EdgeNodes,ElementEdges,\
            BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,\
                ExactE,ExactB,T,dt,theta)
        if task == 'GI':
            Bh,Eh,Berror,Eerror = GISolver(J,Nodes,EdgeNodes,ElementEdges,\
            BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,\
                ExactE,ExactB,T,dt,theta)  
        print('Computed Numerical Solution')
        QuadElectricErr[i] = Eerror
        QuadMagneticErr[i] = Berror
        #    divB                         = DivMat.dot(Bh)
        #    DivErr                       = 0
        #    NEl                          = len(ElementEdges)
        #    for j in range(NEl):
        #        Element                = ElementEdges[j]
        #        Ori                    = Orientations[j]
        #        xP,yP,A,Vertices,Edges = Centroid(Element,EdgeNodes,Nodes,Ori)
        #        DivErr                 = DivErr + A*(divB[j]**2)
        #    #DivErr            = math.sqrt(DivErr)
        #    DivQuadErr[i] = DivErr
        i = i+1
        #with open('Quad'+task+'.txt', "wb") as fp:
        #    pickle.dump((QuadElectricErr,QuadMagneticErr),fp)
    print('Quad Electric = '+str(QuadElectricErr))
    print('Quad Magnetic = '+str(QuadMagneticErr))
    #print('Quad Divergence err ='+str(DivQuadErr))
    ###################################################################################################################
    ###################################################################################################################

    ProcessedFiles = ['PTh=0.101015.txt','PTh=0.051886.txt','PTh=0.0251418.txt','PTh=0.0125255.txt',\
                     'PTh=0.0062613.txt']


    h = [0.10101525445522107, 0.05018856132284956, 0.025141822757713456, 0.012525468249897755,\
         0.006261260829309998]
    # ProcessedFiles = ['PTh=0.00313997.txt','PTh=0.00156773.txt']
    # h              = [0.00313997,0.00156773]
    Basis                    = [Poly1,Poly2,Poly]
    T                        = 0.25

    TriangleElectricErr = [0]*len(ProcessedFiles)
    TriangleMagneticErr = [0]*len(ProcessedFiles)
    DivVoronoiErr      = [0]*len(ProcessedFiles)
    i                  = 0
    for Pfile in ProcessedFiles:
        dt = 0.05*h[i]**2
        #dt = h[i]**2
        #dt = h[i]
        print(Pfile)
        Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations = ProcessedMesh(Pfile)
        if task == 'E':
            Bh,Eh,Berror,Eerror = ESolver(J,Nodes,EdgeNodes,ElementEdges,\
                                        BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,ExactE,ExactB,T,dt,theta)
        if task == 'LS':
            Bh,Eh,Berror,Eerror = LSSolver(J,Nodes,EdgeNodes,ElementEdges,\
            BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,\
                ExactE,ExactB,T,dt,theta)
        if task == 'GI':
            Bh,Eh,Berror,Eerror = GISolver(J,Nodes,EdgeNodes,ElementEdges,\
            BoundaryNodes,Orientations,EssentialBoundaryCond,InitialCond,\
                ExactE,ExactB,T,dt,theta)  
        print('Computed Numerical Solution')
        #VisualizeE(Eh,Nodes)
        TriangleElectricErr[i] = Eerror
        TriangleMagneticErr[i] = Berror
        # divB                         = DivMat.dot(Bh)
        # DivErr                       = 0
        # NEl                          = len(ElementEdges)
        # for j in range(NEl):
        #     Element                = ElementEdges[j]
        #     Ori                    = Orientations[j]
        #     xP,yP,A,Vertices,Edges = Centroid(Element,EdgeNodes,Nodes,Ori)
        #     DivErr                 = DivErr + A*(divB[j]**2)
        # #DivErr            = math.sqrt(DivErr)
        # DivVoronoiErr[i] = DivErr
        i=i+1
        with open('Triangle'+task+'.txt', "wb") as fp:
            pickle.dump((TriangleElectricErr,TriangleMagneticErr),fp)
    #with open('FiveEMVoronoi.txt', "wb") as fp:   #Pickling
    #    pickle.dump([FiveVoronoiElectricError,FiveVoronoiMagneticError], fp)

    print('Triangle Electric = '+str(TriangleElectricErr))
    print('Triangle Magnetic = '+str(TriangleMagneticErr))
    #print('Triangle Divergence err ='+str(DivVoronoiErr))