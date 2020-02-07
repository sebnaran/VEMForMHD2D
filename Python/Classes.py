import numpy as np

class Energy(object):

    def __init__(self,Task,L2,L1,R1,R2):
        self.iR1 = R1
        self.iR2 = R2
        self.iL1 = L1
        self.iL2 = L2