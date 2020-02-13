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