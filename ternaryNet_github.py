import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
import os, time
import sys
import nibabel as nib

# uses five-fold cross-validation on customly pre-processed
# NIH pancreas CT dataset 
fold = sys.argv[1]
fold = int(fold)
print(('fold number ',fold))

allcases = np.arange(0,82)

select_test = allcases[82*(fold-1)//5:82*(fold)//5]
select_train = np.setxor1d(allcases,select_test)
print(select_test)
len_train = select_train.shape[0]

# Dice loss, used for validation without epsilon
def dice_coeff(output, label, use_eps):
    eps = Variable(torch.ones(1)*1e-3)
    label_num = output.size(1)
    output = F.softmax(output,dim=1)    
    output2 = output.permute(1,0,2,3).contiguous().view(label_num,-1).cpu()
    label2 = label.view(1,-1).expand(label_num,label.numel())
    nu = torch.arange(0,label_num,1).view(label_num,1).expand(label_num,label.numel()).long()
    label2 = Variable((label2==nu).float())
    #eps = 1e3#torch.min(,1e-3)
    #eps = (torch.mean(label2,dim=1)*1e-2)
    intersect2 = 2.0*torch.mean(label2*output2,dim=1)+eps*use_eps
    union2 = torch.mean(label2,dim=1)+torch.mean(output2,dim=1)+eps*use_eps
    dice = intersect2/union2
    return dice


#define image dimensions and read data with segmentations
H = 176
W = 110
D = 126
num = 82

imgs = torch.Tensor(num,H,W,D)

for i in range(1,num+1):

    filenii = 'pat' + str(format(i, '04d')) + '.nii.gz'
    img_filename = os.path.join('~/PancreasCT/',filenii)
    img = nib.load(img_filename)
    data_img = img.get_data()
    img1 = torch.from_numpy(data_img.astype('float32')).contiguous().view(1,H,W,D)
    imgs[i-1,:,:,:] = img1
    

segs = torch.Tensor(num,H,W,D)

for i in range(1,num+1):
    filenii = 'label' + str(format(i, '04d')) + '.nii.gz'
    #/share/data_zoe1/heinrich/deedsBCV
    img_filename = os.path.join('~/PancreasCT/',filenii)
    img_nii = nib.load(img_filename)
    data_seg = img_nii.get_data()
    seg_input = torch.from_numpy(data_seg.astype('int64')).contiguous().view(1,H,W,D)
    segs[i-1,:,:,:] = seg_input

# our proposed activation and some variations
# if beta>1: the soft/continuous function is used 
# if beta > 0 and <1 we ternarise (-1,0,+1) activations
def ternaryTanh(x,beta=2.0):
    m = torch.nn.Tanh()
    if(beta>=1.0):
        y = m((x*beta*2.0-beta))*0.5
        y += -m((-x*beta*2.0-beta))*0.5
    elif(beta==0.0):
        y = torch.sign(x)
    elif(beta<0):
        y = torch.nn.HardTanh(x)
    else:
        y = torch.sign(x)*((torch.abs(x)>beta).float())

    return y


#definition of network-architecture as detailed in paper
class myUNet(nn.Module):

    def __init__(self):
        super(myUNet, self).__init__()
        self.name = 'TBC'
        base = 32
        self.label_num = 2
        #important parameter beta in paper
        #for continuouation of ternarisation / slope
        self.delta = 3.0
        self.avg2 = nn.AvgPool2d(kernel_size=(2,2))

        #input should be stack of 15 slices
        self.c1a = nn.Conv2d(15, base, kernel_size=3, bias=False)
        self.c1b = nn.Conv2d(base, base*2, kernel_size=3, bias=False)
        self.b1a = nn.BatchNorm2d(base); self.b1b = nn.BatchNorm2d(base*2)

        self.c2a = nn.Conv2d(base*2, base*2, kernel_size=3, bias=False)
        self.c2b = nn.Conv2d(base*2, base*4, kernel_size=3, bias=False)
        self.b2a = nn.BatchNorm2d(base*2); self.b2b = nn.BatchNorm2d(base*4)

        self.c3a = nn.Conv2d(base*4, base*4, kernel_size=3, bias=False)
        self.c3b = nn.Conv2d(base*4, base*8, kernel_size=3, bias=False)
        self.b3a = nn.BatchNorm2d(base*4); self.b3b = nn.BatchNorm2d(base*8)

        self.c4a = nn.Conv2d(base*8, base*8, kernel_size=1, bias=False)
        self.c4b = nn.Conv2d(base*8, base*8, kernel_size=1, bias=False) #last contracting layer doesnt double channels
        self.b4a = nn.BatchNorm2d(base*8); self.b4b = nn.BatchNorm2d(base*8)

        self.c5a = nn.Conv2d(base*16, base*8, kernel_size=3, bias=False)
        self.c5b = nn.Conv2d(base*8, base*4, kernel_size=3, bias=False)
        self.b5a = nn.BatchNorm2d(base*8); self.b5b = nn.BatchNorm2d(base*4)

        self.c6a = nn.Conv2d(base*8, base*4, kernel_size=3, bias=False)
        self.c6b = nn.Conv2d(base*4, base*2, kernel_size=3, bias=False)
        self.b6a = nn.BatchNorm2d(base*4); self.b6b = nn.BatchNorm2d(base*2)

        self.c7a = nn.Conv2d(base*4, base*2, kernel_size=3, bias=False)
        self.c7b = nn.Conv2d(base*2, base*2, kernel_size=3, bias=False)
        self.b7a = nn.BatchNorm2d(base*2); self.b7b = nn.BatchNorm2d(base*2);

        self.c8 = nn.Conv1d(base*2, self.label_num, kernel_size=1)

    def forward(self, T): 
        B, C, H, W = T.size()

        T1 = F.pad(T,(31,31,30,30)) 
        S1a = ternaryTanh(self.b1a(self.c1a(T1)),self.delta)
        S1b = ternaryTanh(self.b1b(self.c1b(S1a)),self.delta)
        S2a = ternaryTanh(self.b2a(self.c2a(self.avg2(S1b))),self.delta)
        S2b = ternaryTanh(self.b2b(self.c2b(S2a)),self.delta)
        S3a = ternaryTanh(self.b3a(self.c3a(self.avg2(S2b))),self.delta)
        S3b = ternaryTanh(self.b3b(self.c3b(S3a)),self.delta)
        S4a = ternaryTanh(self.b4a(self.c4a(self.avg2(S3b))),self.delta)
        S4a_ = self.c4b(S4a)
        S4b = ternaryTanh(self.b4b(S4a_),self.delta)

        S4uc = torch.cat((F.upsample(S4b,scale_factor=2,mode='bilinear'),S3b),dim=1)
        S5a = ternaryTanh(self.b5a(self.c5a(S4uc)),self.delta)
        S5b = ternaryTanh(self.b5b(self.c5b(S5a)),self.delta)
        S5uc = torch.cat((F.upsample(S5b,scale_factor=2,mode='bilinear'),F.pad(S2b,(-8,-8,-8,-8))),dim=1)
        S6a = ternaryTanh(self.b6a(self.c6a(S5uc)),self.delta)
        S6b = ternaryTanh(self.b6b(self.c6b(S6a)),self.delta)
        S6uc = torch.cat((F.upsample(S6b,scale_factor=2,mode='bilinear'),F.pad(S1b,(-24,-24,-24,-24))),dim=1)
        S7a = ternaryTanh(self.b7a(self.c7a(S6uc)),self.delta)
        S7b = F.relu(self.b7b(self.c7b(S7a)))
        S8 = self.c8(S7b.view(B,S7b.size(1),-1)).view(B,self.label_num,H+4,W+6)

        return F.pad(S8,(-3,-3,-2,-2)), S4a_


#Xavier intialisation
def weights_init(m):
    if isinstance(m, nn.Conv1d):
        init.xavier_normal(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0.0)
    if isinstance(m, nn.Conv2d):
        init.xavier_normal(m.weight, gain=np.sqrt(2))
        #init.constant(m.bias, 0.0)
    if isinstance(m, nn.Linear):
        init.xavier_normal(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0.0)

# ternary weight approximation according to https://arxiv.org/abs/1605.04711
def approx_weights(w_in):
    a,b,c,d = w_in.size()
    delta = 0.7*torch.mean(torch.mean(torch.mean(torch.abs(w_in),dim=3),dim=2),dim=1).view(-1,1,1,1)
    alpha = torch.abs(w_in)*(torch.abs(w_in)>delta).float()
    alpha = (torch.sum(torch.sum(torch.sum(alpha,dim=3),dim=2),dim=1)  \
    /torch.sum(torch.sum(torch.sum((alpha>0).float(),dim=3),dim=2),dim=1)).view(-1,1,1,1)
    w_out = -(w_in<-delta).float()*alpha + (w_in>delta).float()*alpha
    return w_out





#start the training
net = myUNet()
net.apply(weights_init) # apply xavier weight init

net.cuda()
#loss criterion and optimizer 
lossweight = torch.zeros(2) #will be weighted
lossweight = lossweight.cuda()
lossweight[0] = 0.5
lossweight[1] = 2.5

criterion = nn.CrossEntropyLoss(lossweight)#
optimizer = optim.Adam(net.parameters(), lr=0.0025)
def exp_lr_scheduler(optimizer, epoch, lr_decay=0.9, lr_decay_epoch=1):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if epoch % lr_decay_epoch:
        return optimizer
    
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    return optimizer


run_loss = np.zeros(40)
time_all = 0.0
run_count = np.zeros((40,150))
run_dice = np.zeros((40,4))
weight_approx = True #ternarise weights

#sequence of betas
betas = torch.linspace(3.0,8.0,40)

#training epochs
for epoch in range(40):
    net.beta = betas[epoch] #forward path always continuous

    print(('name',net.name,'beta',net.beta))
    
    net.cuda()
    optimizer=exp_lr_scheduler(optimizer, epoch)
    run_loss[epoch] = 0.0
    idx_epoch = torch.randperm(len_train*96-(len_train*96)%10).view(10,-1)
    #mini-batches
    net.train()
    use_eps = Variable(torch.ones(1))
    t0 = time.time()
    for iter in range(150):
        
        idx_scan = select_train[idx_epoch[:,iter]%len_train]
        idx_slice = idx_epoch[:,iter]/65
            
        #store high precision weights for later
        stored_approx_c1b = net.c1b.weight.data
        stored_approx_c2a = net.c2a.weight.data
        stored_approx_c2b = net.c2b.weight.data
        stored_approx_c3a = net.c3a.weight.data
        stored_approx_c3b = net.c3b.weight.data
        stored_approx_c4a = net.c4a.weight.data
        stored_approx_c4b = net.c4b.weight.data
        stored_approx_c5a = net.c5a.weight.data
        stored_approx_c5b = net.c5b.weight.data
        stored_approx_c6a = net.c6a.weight.data
        stored_approx_c6b = net.c6b.weight.data
        stored_approx_c7a = net.c7a.weight.data
        stored_approx_c7b = net.c7b.weight.data

        if(weight_approx):
        #assign approximate weights
            net.c1b.weight.data = approx_weights(stored_approx_c1b)
            net.c2a.weight.data = approx_weights(stored_approx_c2a)
            net.c2b.weight.data = approx_weights(stored_approx_c2b)
            net.c3a.weight.data = approx_weights(stored_approx_c3a)
            net.c3b.weight.data = approx_weights(stored_approx_c3b)
            net.c4a.weight.data = approx_weights(stored_approx_c4a)
            net.c4b.weight.data = approx_weights(stored_approx_c4b)
            net.c5a.weight.data = approx_weights(stored_approx_c5a)
            net.c5b.weight.data = approx_weights(stored_approx_c5b)
            net.c6a.weight.data = approx_weights(stored_approx_c6a)
            net.c6b.weight.data = approx_weights(stored_approx_c6b)
            net.c7a.weight.data = approx_weights(stored_approx_c7a)
            net.c7b.weight.data = approx_weights(stored_approx_c7b)

 
        optimizer.zero_grad() 
        #forward path and loss
        T = torch.zeros(10,15,H,W)
        L = torch.zeros(10,H,W).long()
        for i in range(10):
            T[i,:,:,:]=imgs[idx_scan[i],:,:,idx_slice[i]:idx_slice[i]+30:2].permute(2,0,1)
            L[i,:,:]=segs[idx_scan[i],:,:,idx_slice[i]+15]
        label_hist = torch.histc(L.view(-1).float(), bins=2, min=0, max=1).long()
        run_count[epoch,iter]=torch.sum(L==1)
        output = net(Variable(T).cuda())
        loss = criterion(output[0], Variable(L).cuda())
            
    
        loss.backward()
        #before grad update - reassign full precision weights
        net.c1b.weight.data = stored_approx_c1b 
        net.c2a.weight.data = stored_approx_c2a 
        net.c2b.weight.data = stored_approx_c2b
        net.c3a.weight.data = stored_approx_c3a 
        net.c3b.weight.data = stored_approx_c3b
        net.c4a.weight.data = stored_approx_c4a 
        net.c4b.weight.data = stored_approx_c4b
        net.c5a.weight.data = stored_approx_c5a 
        net.c5b.weight.data = stored_approx_c5b
        net.c6a.weight.data = stored_approx_c6a 
        net.c6b.weight.data = stored_approx_c6b
        net.c7a.weight.data = stored_approx_c7a 
        net.c7b.weight.data = stored_approx_c7b

        optimizer.step()

        run_loss[epoch] += loss.data[0]
    print((epoch,run_loss[epoch]*0.125,'time: ',time.time()-t0))
    

    net.eval()
    
    #store high precision weights for later
    stored_approx_c1b = net.c1b.weight.data
    stored_approx_c2a = net.c2a.weight.data
    stored_approx_c2b = net.c2b.weight.data
    stored_approx_c3a = net.c3a.weight.data
    stored_approx_c3b = net.c3b.weight.data
    stored_approx_c4a = net.c4a.weight.data
    stored_approx_c4b = net.c4b.weight.data
    stored_approx_c5a = net.c5a.weight.data
    stored_approx_c5b = net.c5b.weight.data
    stored_approx_c6a = net.c6a.weight.data
    stored_approx_c6b = net.c6b.weight.data
    stored_approx_c7a = net.c7a.weight.data
    stored_approx_c7b = net.c7b.weight.data

    if(weight_approx):
        #assign approximate weights
        net.c1b.weight.data = approx_weights(stored_approx_c1b)
        net.c2a.weight.data = approx_weights(stored_approx_c2a)
        net.c2b.weight.data = approx_weights(stored_approx_c2b)
        net.c3a.weight.data = approx_weights(stored_approx_c3a)
        net.c3b.weight.data = approx_weights(stored_approx_c3b)
        net.c4a.weight.data = approx_weights(stored_approx_c4a)
        net.c4b.weight.data = approx_weights(stored_approx_c4b)
        net.c5a.weight.data = approx_weights(stored_approx_c5a)
        net.c5b.weight.data = approx_weights(stored_approx_c5b)
        net.c6a.weight.data = approx_weights(stored_approx_c6a)
        net.c6b.weight.data = approx_weights(stored_approx_c6b)
        net.c7a.weight.data = approx_weights(stored_approx_c7a)
        net.c7b.weight.data = approx_weights(stored_approx_c7b)


    T = torch.zeros(16,15,H,W)
    L = torch.zeros(16,H,W).long()
    idx_slice = np.arange(20,100,5)
    dice_all = np.zeros((82,4))

    #we first test the network without ternarised activations
    use_eps = Variable(torch.zeros(1))
    net.beta = betas[epoch]

    for nu_test in range (select_test.shape[0]):
        test_case = select_test[nu_test]

        for i in range(16):
            #Tl[i,:,:,:] = imgs[16,:,:,idx_slice[i]+11:idx_slice[i]+20].permute(2,0,1)
            T[i,:,:,:]=imgs[test_case,:,:,idx_slice[i]:idx_slice[i]+30:2].permute(2,0,1)
            L[i,:,:]=segs[test_case,:,:,idx_slice[i]+15]

        output = net(Variable(T).cuda())
        dice_val = dice_coeff(output[0]*1000.0,L,use_eps)
        dice_all[test_case,:2] = dice_val.data.numpy()

    print((output[1].view(-1).size(),np.unique((torch.abs(output[1].data.cpu().view(-1))).numpy()).shape))
        

    #and then activate the step that replaces activations by (-1,0,+1)
    #these are our main results
    net.beta = 0.5
    
    for nu_test in range (select_test.shape[0]):
        test_case = select_test[nu_test]

        for i in range(16):
            #Tl[i,:,:,:] = imgs[16,:,:,idx_slice[i]+11:idx_slice[i]+20].permute(2,0,1)
            T[i,:,:,:]=imgs[test_case,:,:,idx_slice[i]:idx_slice[i]+30:2].permute(2,0,1)
            L[i,:,:]=segs[test_case,:,:,idx_slice[i]+15]

        output = net(Variable(T).cuda())
        dice_val = dice_coeff(output[0]*1000.0,L,use_eps)
        dice_all[test_case,2:] = dice_val.data.numpy()

    print((output[1].view(-1).size(),np.unique((torch.abs(output[1].data.cpu().view(-1))).numpy()).shape))
        

    #store some numbers
    run_dice[epoch,:] = (np.mean(dice_all[select_test,:],axis=0))
    np.savetxt('/home/heinrich/notebooks/ijcars_unet_ternary_main'+str(fold)+'_'+str(epoch)+'.txt',dice_all)

    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    print(('dice_val',run_dice[epoch,:]))#dice_val.view(1,-1).data.numpy()))
    net.c1b.weight.data = stored_approx_c1b 
    net.c2a.weight.data = stored_approx_c2a 
    net.c2b.weight.data = stored_approx_c2b
    net.c3a.weight.data = stored_approx_c3a 
    net.c3b.weight.data = stored_approx_c3b
    net.c4a.weight.data = stored_approx_c4a 
    net.c4b.weight.data = stored_approx_c4b
    net.c5a.weight.data = stored_approx_c5a 
    net.c5b.weight.data = stored_approx_c5b
    net.c6a.weight.data = stored_approx_c6a 
    net.c6b.weight.data = stored_approx_c6b
    net.c7a.weight.data = stored_approx_c7a 
    net.c7b.weight.data = stored_approx_c7b
    
    net.cpu()
    torch.save(net.state_dict(), 'ijcars_unet_ternary_main'+str(fold)+'.pth')
    
#store final weights and plot loss curve
net.cpu()
torch.save(net.state_dict(), 'ijcars_unet_ternary_main'+str(fold)+'.pth')
    
plt.plot(0.01*run_loss[:])
plt.plot(run_dice[:,1])
plt.plot(run_dice[:,3])
plt.savefig(ijcars_unet_ternary_main'+str(fold)+'.png')

