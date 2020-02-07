import numpy as np

class Energy(object):

    def __init__(self,MV,ME,MJ):

        self.iMV  = MV
        self.iME  = ME
        self.iMJ  = MJ
    
    def ComputeCoeffs(self,theta,dt,T):
        curl = primcurl(EdgeNodes,Nodes)
        D    = lil_matrix((len(Nodes),len(Nodes)))
        for i in InternalNodes:
            D[i,i]=1
        D = D.tocsr()

        Aprime = MV+theta*dt*( ( np.transpose(curl) ).dot(ME)+MJ ).dot(curl)#MV.dot(MJ) ).dot(curl)
        Aprime = D.dot(Aprime)
        A      = lil_matrix((NumberInternalNodes,NumberInternalNodes))
        
        for i in range(NumberInternalNodes):
            A[i,:] = Aprime[InternalNodes[i],InternalNodes]
        A = A.tocsr()
    
        b = np.transpose(curl).dot(ME)+MJ#+MV.dot(MJ)
        b = D.dot(b)

        Bh   = HighOrder7projE(InitialCond,EdgeNodes,Nodes)
        Bh   = np.transpose(Bh)[0]
        RHS1 = Bh.dot(ME.dot(Bh))

        Eh         = np.zeros(len(Nodes))
        EhInterior = np.zeros(len(Nodes))
        EhBoundary = np.zeros(len(Nodes))
        step       = 1
        for t in time[0:len(time)-1]:

            for NodeNumber in BoundaryNodes:
                Node = Nodes[NodeNumber]
                EhBoundary[NodeNumber] = EssentialBoundaryCond(Node[0],Node[1],t+theta*dt)

            RHS2 = RHS2+\
                   (beta**step)*gamma*dt*\
                   ( EhBoundary.dot(MV.dot(EhBoundary))+\
                   (curl.dot(EhBoundary)).dot(ME.dot(curl.dot(EhBoundary))) )

            W1 = b.dot(Bh)
            W2 = Aprime.dot(EhBoundary)

            EhInterior[InternalNodes] = spsolve(A,W1[InternalNodes]-W2[InternalNodes])

            Eh   = EhInterior+EhBoundary
            LHS2 = LHS2+0.5*gamma*dt(beta**(step))*Eh.dot(MV.dot(Eh))

            Bh   = Bh-dt*curl.dot(Eh)
            LHS1 = (beta**step)*Bh.dot(ME.dot(Bh))

            LHS[step] = LHS1+LHS2
            RHS[step] = RHS1+RHS2
            step      = step+1
