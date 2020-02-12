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
        k = k+1
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
#for Element in ElementEdges:
# if True:
#     n                      = len(Element)
#     #Ori                    = Orientations[k]
#     k                      = k+1
#     OrVert,OrEdg           = StandardElement(Element,EdgeNodes,Nodes,Ori)
#     xP,yP,A,Vertices,Edges = Centroid(Element,EdgeNodes,Nodes,Ori)

#     ElNodes  = OrVert[0:n] 
#     xs       = 0
#     ys       = 0
#     for node in ElNodes:
#         xs = xs+node[0]
#         ys = ys+node[1]
    
#     xs = xs/n
#     ys = ys/n
#     Areas = [0]*n
#     #2*Area = Ax(By - Cy) + Bx(Cy - Ay) + Cx(Ay - By)
#     #A=starred B = kth C=k+1th
#     xk         = ElNodes[n-1][0]
#     yk         = ElNodes[n-1][1]
#     xkp1       = ElNodes[0][0]
#     ykp1       = ElNodes[0][1]
#     Areas[n-1] = abs(xs*(yk-ykp1)+xk*(ykp1-xs)+xkp1*(ys-yk))/2

#     for i in range(n-1):
#         xk       = ElNodes[i][0]
#         yk       = ElNodes[i][1]
#         xkp1     = ElNodes[i+1][0]
#         ykp1     = ElNodes[i+1][1]
#         Areas[i] = abs(xs*(yk-ykp1)+xk*(ykp1-xs)+xkp1*(ys-yk))/2
#     w    = [0]*n
#     w[0] = ( (n+1)*(Areas[0]+Areas[n-1])+sum(Areas[1:n-1]) )/(3*n*A)
#     cx   = w[0]*ElNodes[0][0]
#     cy   = w[0]*ElNodes[0][1]
#     print(Areas)
    
#     for i in range(1,n):
#         print(i)
#         A1         = Areas[0:i-1]
#         A2         = Areas[i+1:n]
#         print(A1)
#         print(A2)
#         s          = sum(A1)
#         p          = sum(A2)
#         w[i] = ( (n+1)*(Areas[i]+Areas[i-1])+s+p )/(3*n*A)
#         cx = cx+w[i]*ElNodes[i][0]
#         cy = cy+w[i]*ElNodes[i][1]

#     print(cx)
#     print(xP)
#     print(yP)
#     print(cy)
#     print(w)
#     print(sum(w))
#     print(A)
#     print(Areas)

