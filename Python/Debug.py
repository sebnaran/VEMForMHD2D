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

# Basis = [Poly1,Poly2,Poly]
# ProcessedFiles = ['PVh=0.128037.txt','PVh=0.0677285.txt','PVh=0.0345033.txt','PVh=0.0174767.txt',\
#                  'PVh=0.0087872.txt']
# Pfile          = ProcessedFiles[0]
# Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations = ProcessedMesh(Pfile)
# Element        = ElementEdges[2]
# Ori            = Orientations[2]
# LocME,LocMV,LocMJ,Edges,B = PieceWiseLocalMEWEMVWV(J,Basis,Element,EdgeNodes,Nodes,Ori)
# #This trial mesh is the square with vertices at [pm 1, pm 1]
# # Nodes     = [[-1,-1],[1,-1],[1,1],[-1,1]]
# # EdgeNodes = [[0,1],[1,2],[2,3],[3,0]]
# # Element   = [0,1,2,3]
# # Ori       = [1,1,1,1]

# # ME,MV,MJ,Edges,B = PieceWiseLocalMEWEMVWV(J,Basis,Element,EdgeNodes,Nodes,Ori)
# # Ar               = np.array([1,1,1,1])
# # print(Ar.dot(MV.dot(Ar)))
# print(B)
# places = [1,2]
# with open('listofplaces.txt','w') as file:
#     file.writelines('[')
#     for e in range(len(places)):
#         if e == 0:
#             file.writelines(str(places[e]))
#         else:
#             file.writelines(',')
#             file.writelines(str(places[e]))
        
#     file.writelines('];')
#     file.write('\n')
x = [1,2,3,4]
y = [32,4,6]
with open('love.txt', "wb") as fp:
    pickle.dump((x,y),fp)
    #pickle.dump(y,fp)

with open('love.txt',"rb") as fb:
    a,s = pickle.load(fb)

print(a)
print(s)