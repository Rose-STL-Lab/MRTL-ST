import torch


class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self, name)
            # return getattr(self.module, name)



class Full(torch.nn.Module):
    def __init__(self, a_dims, b_dims, c_dims, counts):
        super().__init__()
#         a_dims +=1

        self.a_dims = a_dims
        self.b_dims = b_dims
        self.c_dims = c_dims
        self.W = torch.nn.Parameter(torch.randn((a_dims, 4 * (*b_dims), *c_dims)),
                                    requires_grad=True)
        print('W shape = ' + str(self.W.shape))
        # self.b = torch.nn.Parameter(torch.ones(a_dims) * np.log(counts[1] / (counts[0])), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros(a_dims), requires_grad=True)

    def forward(self, a, bh_pos, def_pos):
        # Only some defenders in defender box
#         print(self.W[a.long(), bh_pos[:, 0].long(),bh_pos[:, 1].long(), :, :].size)
#         print(self.W.size())
#         print((a.long(), bh_pos[:, 0].long(),bh_pos[:, 1].long()))


        (temp1, temp2) = (bh_pos[:, 0].long(), bh_pos[:, 1].long())
        temp3 = a.long()

        
        # print("bh_pos[:, 0].long(): " + str(bh_pos[:, 0].long()))
        # print("bh_pos[:, 0][0].long(): " + str(bh_pos[:, 0][0].long()))
        # print("self.W.size()[1]: " + str(self.W.size()[1]))

        # if bh_pos[:, 0].long() >= self.W.size()[1]:
        #     temp1 = self.W.size()[1]-1
        # if bh_pos[:, 1].long() >= self.W.size()[2]:
        #     temp2 = self.W.size()[2]-1
#         print('test')
        one1 = torch.ones_like(temp1) * (self.W.size()[1]-1)
        temp1 = torch.where(temp1 >= self.W.size()[1], one1, temp1)
        one2 = torch.ones_like(temp2) * (self.W.size()[2]-1)
        temp2 = torch.where(temp2 >= self.W.size()[2], one2, temp2)
        one3 = torch.ones_like(temp3) * (self.W.size()[0]-1)

        temp3 = torch.where(temp3 >= self.W.size()[0], one3, temp3)
#         print(temp1,temp2)
#         print(self.W.size())
#         print(a.long())
        
            
#         print("Dim match")
#         print(self.W[a.long(), temp1,temp2, :, :].size())
#         print(def_pos.float().size())

#         print("End")
        # print(self.W[a.long(), temp1,temp2, :, :])
            


        out = torch.einsum(
            'bcd,bcd->b', self.W[temp3, temp1,temp2, :, :], def_pos.float())
        return out.add_(self.b[temp3])


class Low(torch.nn.Module):
    def __init__(self, a_dims, b_dims, c_dims, K, counts):
        super().__init__()
#         a_dims +=1
        self.a_dims = a_dims
        self.b_dims = b_dims
        self.c_dims = c_dims
        self.K = K

        self.A = torch.nn.Parameter(torch.randn((a_dims, K)),
                                    requires_grad=True)
        self.B = torch.nn.Parameter(torch.randn(*b_dims, K),
                                    requires_grad=True)
        self.C = torch.nn.Parameter(torch.randn(*c_dims, K),
                                    requires_grad=True)


        # self.b = torch.nn.Parameter(torch.ones(a_dims) * np.log(counts[1] / (counts[0])), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros(a_dims), requires_grad=True)

    def forward(self, a, bh_pos, def_pos):
#         print("Dims")
#         print(self.A[a.long(), :].size())
#         print(bh_pos[:, 0].long())
#         print(bh_pos[:, 1].long())
        (temp1, temp2) = (bh_pos[:, 0].long(), bh_pos[:, 1].long())
        temp3 = a.long()
        # if bh_pos[:, 0].long() >= self.B.size()[0]:
        #     temp1 = self.B.size()[0]-1
        # if bh_pos[:, 1].long() >= self.B.size()[1]:
        #     temp2 = self.B.size()[1]-1

#         one2 = torch.ones_like(temp2) * self.W.size()[2]-1
#         temp2 = torch.where(temp2 > self.W.size()[2], one2, temp2)
        one1 = torch.ones_like(temp1) * (self.B.size()[0]-1)
        temp1 = torch.where(temp1 >= self.B.size()[0], one1, temp1)
        one2 = torch.ones_like(temp2) * (self.B.size()[1]-1)
        temp2 = torch.where(temp2 >= self.B.size()[1], one2, temp2)
        one3 = torch.ones_like(temp3) * (self.A.size()[0]-1)

        temp3 = torch.where(temp3 >= self.A.size()[0], one3, temp3)
            
#         print(self.B.size())

#         print(self.B[temp1, temp2, :])
#         print(self.B[temp1,temp2, :].size())
#         print(def_pos.size())
#         print(self.C.size())
#         print(torch.einsum('bcd,cde->be', def_pos.float(), self.C).size())
#         print("End")
#         print(self.A.size())
#         print(a.long())
#         print(self.A[temp3, :])
#         print(temp1)
#         print(temp2)
#         print(self.B.size())

#         print(self.B[temp1,temp2, :])
   
        # Tensor multiply def_pos and self.C to get sum for all defenders, and then sum over latent factors
        out = (self.A[temp3, :] *
               self.B[temp1,temp2, :] *
               torch.einsum('bcd,cde->be', def_pos.float(), self.C)).sum(1)
        
        return out.add_(self.b[temp3])

    def constrain(self):
        # Clamp weights
        self.A.clamp_min_(min=0)
        self.B.clamp_min_(min=0)
        # self.C.clamp_max_(max=0)
