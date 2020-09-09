import torch
from torch.nn import Module, BCELoss

class CumulativeLoss(Module):
    def __init__(self, beta):
        super().__init__()
        self.criterion = BCELoss()
        self.beta = beta

    def forward(self, label, final_op, aux_op=None):
        fin_loss = self.criterion(final_op, label)
        if aux_op is not None:
            aux_label = torch.ones((label.shape[0], 196)).type(torch.FloatTensor).cuda()
            for i in range(label.shape[0]):
                aux_label[i] = aux_label[i]*label[i]
            aux_loss = self.criterion(aux_op, aux_label)
            cumulative_loss = self.beta*aux_loss + (1-self.beta)*fin_loss
        else: 
            cumulative_loss = fin_loss

        return cumulative_loss
    


if __name__ == "__main__":
    label = torch.ones([4, 1], dtype=torch.float).cuda()
    fin_op = torch.sigmoid(torch.randn([4, 1], dtype=torch.float)).cuda()

    aux_op = torch.sigmoid(torch.randn([4, 1, 14, 14], dtype=torch.float))
    aux_op = aux_op.view(aux_op.shape[0], -1).cuda()
    print(aux_op.shape)

    loss = CumulativeLoss(0.5)
    fin_loss = loss(label, fin_op, aux_op)
    print(fin_loss)