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
def test_convexcoeffs1():
    Nodes     = [[-1,-1],[1,-1],[1,1],[-1,1]]
    EdgeNodes = [[0,1],[1,2],[2,3],[3,0]]
    Element   = [0,1,2,3]
    Ori       = [1,1,1,1]
    w,Areas   = convexcoeffs(Element,EdgeNodes,Nodes,Ori)
    cx        = 0
    cy        = 0
    for i in range(4):
        cx = cx +w[i]*Nodes[i][0]
        cx = cx +w[i]*Nodes[i][1]
    assert abs(sum(Areas)-4) <  0.001
    assert cx                == 0
    assert cy                == 0
    assert sum(w)            == 1

def test_coeffs2():
    Pfile = 'PVh=0.0677285.txt'
    Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations = ProcessedMesh(Pfile)
    k = 0
    for Element in ElementEdges:
        Ori = Orientations[k]
        k   = k+1
        w,Areas      = convexcoeffs(Element,EdgeNodes,Nodes,Ori)
        cx           = 0
        cy           = 0
        OrVert,OrEdg = StandardElement(Element,EdgeNodes,Nodes,Ori)
        n        = len(Element)
        ElNodes  = OrVert[0:n]
        i = 0
        for Node in ElNodes:
            cx = cx + w[i]*Node[0]
            cy = cy + w[i]*Node[1]
            i  = i+1

        xP,yP,A,Vertices,Edges = Centroid(Element,EdgeNodes,Nodes,Ori)

        assert abs(sum(Areas)-A) < 0.0001
        assert abs(sum(w)-1)     < 0.0001
        assert abs(cx-xP)        < 0.0001
        assert abs(cy-yP)        < 0.0001

def test_reconstruct():
    def testB(x,y):
        return 1,1
    Pfile = 'HartPVh=0.0124216.txt'
    Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations = ProcessedMesh(Pfile)
    Bh = HighOrder7projE(testB,EdgeNodes,Nodes)
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

    #print(Bx)
    #print(By)
    assert (Bx[10]-1)<10^(-3)
    assert (By[10]-1)<10^(-3)