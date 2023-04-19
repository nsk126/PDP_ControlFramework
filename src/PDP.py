import casadi as ca

class PDPmethod:
    def __init__(self):
        pass
        
    def setAuxVar(self, auxvar):
        self.auxvar = auxvar
        self.n_auxvar = self.auxvar.numel()