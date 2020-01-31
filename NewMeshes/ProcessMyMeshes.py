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
from ../Functions.py import *

Files = ['Vh=0.00442309.txt','Vh=0.00221383.txt','Th=0.00313997.txt','Th=0.00156773.txt','Qh=0.00549451.txt','Qh=0.00275482.txt']

for file in Files:
    print('Making P'+file)
    Nodes,EdgeNodes,ElementEdges,BoundaryNodes = Mesh(file)
    Orientations                               = [0]*len(ElementEdges)
    i                                          = 0
    for Element in ElementEdges:

        Ori             = Orientation(Element,EdgeNodes,Nodes)
        Orientations[i] = Ori
        i               = i+1
    with open('P'+file, "wb") as fp:
        pickle.dump((Nodes,EdgeNodes,ElementEdges,Orientations),fp)