from Classes import Energy
def test_Init():
    L1 = [1]
    R1 = [2]
    L2 = [3]
    R2 = [4]
    
    TestEner = Energy('H1',L2,L1,R1,R2)
    assert TestEner.iR1 == R1
    assert TestEner.iR2 == R2
    assert TestEner.iL1 == L1
    assert TestEner.iL2 == L2
    assert TestEner.Type == 'H1'