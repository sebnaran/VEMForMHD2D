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

Basis = [Poly1,Poly2,Poly]
ProcessedFiles = ['PVh=0.128037.txt','PVh=0.0677285.txt','PVh=0.0345033.txt','PVh=0.0174767.txt',\
                  'PVh=0.0087872.txt']
Pfile          = ProcessedFiles[0]
Nodes,EdgeNodes,ElementEdges,BoundaryNodes,Orientations = ProcessedMesh(Pfile)
Element        = ElementEdges[2]
Ori            = Orientations[2]
#LocME,LocMV,LocMJ,Edges = PieceWiseLocalMEWEMVWV(J,Basis,Element,EdgeNodes,Nodes,Ori)
f = PieceWiseLocalMEWEMVWV(J,Basis,Element,EdgeNodes,Nodes,Ori)