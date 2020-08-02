import torch
from torch.nn import Module, BCELoss

class CumulativeLoss(Module):
    def __init__(self, beta):
        super().__init__()
        self.criterion = BCELoss()
        self.beta = beta

    def forward(self, aux_op, final_op, label):
        aux_loss = self.criterion(aux_op.squeeze(), CumulativeLoss._convertYLabel(label))
        fin_loss = self.criterion(final_op, label)
        
        #cumulative_loss = self.beta*aux_loss + (1-self.beta)*fin_loss
        return fin_loss
    
    def _convertYLabel(y):

        returnY = torch.ones((y.shape[0], 196)).type(torch.FloatTensor)

        for i in range(y.shape[0]):
            returnY[i] = returnY[i]*y[i]

        returnY.cuda() 

if __name__ == "__main__":
    label = torch.ones([4, 1], dtype=torch.float)
    fin_op = torch.sigmoid(torch.randn([4, 1], dtype=torch.float))

    aux_op = torch.sigmoid(torch.randn([4, 1, 14, 14], dtype=torch.float))
    aux_op = aux_op.view(aux_op.shape[0], -1)
    print(aux_op.shape)

    loss = CumulativeLoss(0.5)
    fin_loss = loss(aux_op, fin_op, label)
    print(fin_loss)