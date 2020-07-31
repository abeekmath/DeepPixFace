from torch.nn import Module, BCELoss

class CumulativeLoss(Module):
    def __init__(self, beta):
        super().__init__()
        self.criterion = BCELoss()
        self.beta = beta

    def forward(self, final_op, label):
        #aux_loss = self.criterion(aux_op, label)
        fin_loss = self.criterion(final_op, label)
        
        #cumulative_loss = self.beta*aux_loss + (1-self.beta)*fin_loss
        return fin_loss