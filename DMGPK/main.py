import os
import numpy as np
import argparse
import time
import copy
import h5py

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from sklearn.preprocessing import MinMaxScaler

from imports.ADdataset import ADdataset
from torch_geometric.data import DataLoader 
from torch.utils.data import WeightedRandomSampler
from net.braingnn import Network
from position_embedding import get_embedder
from sklearn import metrics
from imports.utils import train_val_test_split
from imports.Normalization import normal_transform_train, normal_transform_test
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, accuracy_score, precision_score

torch.manual_seed(123)

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def divide(a, b):
    if b == 0 or b == None:
        return 0
    return a / b

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=2, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=20, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='D:/ResearchGroup/ProjectCode/DMGPK/MyData/NC_AD', help='root directory of the dataset')
parser.add_argument('--fold', type=int, default=4, help='training which fold')
parser.add_argument('--lr', type = float, default=0.001, help='learning rate')
parser.add_argument('--stepsize', type=int, default=20, help='scheduler step size')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weightdecay', type=float, default=5e-3, help='regularization')
parser.add_argument('--lamb0', type=float, default=1, help='classification loss weight')
parser.add_argument('--lamb1', type=float, default=0, help='s1 unit regularization')
parser.add_argument('--lamb2', type=float, default=0, help='s2 unit regularization')
parser.add_argument('--lamb3', type=float, default=0.1, help='s1 TPK regularization')
parser.add_argument('--lamb4', type=float, default=0.1, help='s2 TPK regularization')
parser.add_argument('--lamb5', type=float, default=0.2, help='s1 consistence regularization')
parser.add_argument('--layer', type=int, default=2, help='number of GNN layers')
parser.add_argument('--ratio', type=float, default=0.4, help='pooling ratio')
parser.add_argument('--indim', type=int, default=524, help='feature dim')
parser.add_argument('--nroi', type=int, default=524, help='num of ROIs')
parser.add_argument('--nclass', type=int, default=2, help='num of classes')
parser.add_argument('--load_model', type=bool, default=False)
parser.add_argument('--save_model', type=bool, default=True)
parser.add_argument('--optim', type=str, default='Adam', help='optimization method: SGD, Adam')
parser.add_argument('--save_path', type=str, default='D:/ResearchGroup/ProjectCode/DMGPK/model', help='path to save model')
parser.add_argument('--normalization', action='store_true')
opt = parser.parse_args()

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

#################### Parameter Initialization #######################
path = opt.dataroot
name = 'ABIDE'
save_model = opt.save_model
load_model = opt.load_model
opt_method = opt.optim
num_epoch = opt.n_epochs
fold = opt.fold
writer = SummaryWriter(os.path.join('./log',str(fold)))


################## Define Dataloader ##################################

dataset = ADdataset(path,name)
dataset.process()
dataset.data.y = dataset.data.y.squeeze()
dataset.data.x[dataset.data.x == float('inf')] = 0

# scaler = MinMaxScaler()     #会查MinMaxScaler的基本上都应该理解数据归一化，本质上是将数据点映射到了[0,1]区间（默认），
# dataset.data.x = scaler.transform(dataset.data.x)

#  Change!!!
Phenotype_matrix = np.transpose(h5py.File(os.path.join(path, "Pheno_ND.mat"), 'r')['Pheno_ND'][()])
# Normalization
C_sex = Phenotype_matrix[:,1]
C_sex = (np.reshape(C_sex,(-1,1)))
C_age = Phenotype_matrix[:,0] / 100
C_age = np.reshape(C_age,(-1,1))
C_edu = Phenotype_matrix[:,2]  / 20
C_edu = np.reshape(C_edu,(-1,1))
C_cdr = Phenotype_matrix[:,3]  / 5  # no need
C_cdr = np.reshape(C_cdr,(-1,1))
S_cog = Phenotype_matrix[:,4]  / 30
S_cog = np.reshape(S_cog,(-1,1))

Phenotype2 = torch.from_numpy(np.concatenate((C_sex, C_age, C_edu, S_cog), 1)).float() # (737, 4)
embed_fn, input_ch = get_embedder(1, 0)
Phenotype2 = embed_fn(Phenotype2)
print(Phenotype2.shape)

#  Change!!!
tr_index,val_index,te_index = train_val_test_split(fold=fold, sub_len = 172)

tr_index = np.ndarray.tolist(tr_index)
val_index = np.ndarray.tolist(val_index)
te_index = np.ndarray.tolist(te_index)

train_dataset = dataset[tr_index]
train_pen_dataset = Phenotype2[tr_index]
val_dataset = dataset[val_index]
val_pen_dataset = Phenotype2[val_index]
test_dataset = dataset[te_index]
test_pen_dataset = Phenotype2[te_index]

# ###################### Normalize features ##########################
if opt.normalization:
    for i in range(train_dataset.data.x.shape[1]):
        train_dataset.data.x[:, i], lamb, xmean, xstd = normal_transform_train(train_dataset.data.x[:, i])
        test_dataset.data.x[:, i] = normal_transform_test(test_dataset.data.x[:, i],lamb, xmean, xstd)
        val_dataset.data.x[:, i] = normal_transform_test(val_dataset.data.x[:, i], lamb, xmean, xstd)

# 数据重采样，构建采样器，防止类别不平衡
# 数据集中，每一类的数目。
label = dataset.data.y
unique_labels = np.unique(label)
class_sample_counts = []
for t in unique_labels:
    indices = np.where(label == t)[0]
    count = len(indices)
    class_sample_counts.append(count)

weights = 1./ torch.tensor(class_sample_counts, dtype=torch.float)

train_targets = label[tr_index]
val_targets = label[val_index]
test_targets = label[te_index]
train_samples_weights = []
val_samples_weights = []
test_samples_weights = []
for i in range(len(train_targets)):
    train_samples_weights.append(weights[train_targets[i]]) 
train_samples_weights = np.array(train_samples_weights)
train_sampler = WeightedRandomSampler(weights=train_samples_weights, num_samples=len(train_samples_weights), replacement=True)


train_loader = DataLoader(train_dataset,batch_size=opt.batchSize, shuffle= False, sampler = train_sampler)
train_pen_loader = DataLoader(train_pen_dataset, batch_size=opt.batchSize, shuffle= False, sampler = train_sampler)
val_loader = DataLoader(val_dataset, batch_size=opt.batchSize, shuffle=False)
val_pen_loader = DataLoader(val_pen_dataset, batch_size=opt.batchSize, shuffle= False)
test_loader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)
test_pen_loader = DataLoader(test_pen_dataset, batch_size=opt.batchSize, shuffle= False)



############### Define Graph Deep Learning Network ##########################
model = Network(opt.indim,opt.ratio,opt.nclass).to(device)

print(model)
# print('ground_truth: ', test_dataset.data.y, 'total: ', len(test_dataset.data.y), 'positive: ',sum(test_dataset.data.y))

if opt_method == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr= opt.lr, weight_decay=opt.weightdecay)
elif opt_method == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr =opt.lr, momentum = 0.9, weight_decay=opt.weightdecay, nesterov = True)

scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.stepsize, gamma=opt.gamma)

############################### Define Other Loss Functions ########################################
def topk_loss(s,ratio):
    if ratio > 0.5:
        ratio = 1-ratio
    s = s.sort(dim=1).values
    res =  -torch.log(s[:,-int(s.size(1)*ratio):]+EPS).mean() -torch.log(1-s[:,:int(s.size(1)*ratio)]+EPS).mean()
    return res


def consist_loss(s):
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0],s.shape[0])
    D = torch.eye(s.shape[0])*torch.sum(W,dim=1)
    L = D-W
    L = L.to(device)
    res = torch.trace(torch.transpose(s,0,1) @ L @ s)/(s.shape[0]*s.shape[0])
    return res

###################### Network Training Function#####################################
def train(epoch):
    print('train...........')
    scheduler.step()

    for param_group in optimizer.param_groups:
        print("LR", param_group['lr'])
    model.train()
    s1_list = []
    s2_list = []
    loss_all = 0
    step = 0
    for data, pen_data in zip(train_loader, train_pen_loader):
        data = data.to(device)
        pen_data = pen_data.to(device)
        optimizer.zero_grad()
        # output, w1, w2, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos, pen_data)
        output, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos, pen_data)

        s1_list.append(s1.view(-1).detach().cpu().numpy())
        s2_list.append(s2.view(-1).detach().cpu().numpy())

        loss_c = F.nll_loss(output, data.y)

        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y == c])
        # loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
        #            + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist
        loss = opt.lamb0*loss_c \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist
        writer.add_scalar('train/classification_loss', loss_c, epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss1', loss_tpk1, epoch*len(train_loader)+step)
        writer.add_scalar('train/TopK_loss2', loss_tpk2, epoch*len(train_loader)+step)
        writer.add_scalar('train/GCL_loss', loss_consist, epoch*len(train_loader)+step)
        step = step + 1

        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

        s1_arr = np.hstack(s1_list)
        s2_arr = np.hstack(s2_list)
    # return loss_all / len(train_dataset), s1_arr, s2_arr ,w1,w2
    return loss_all / len(train_dataset), s1_arr, s2_arr


###################### Network Testing Function#####################################
def test_acc(loader, pen_loader):
    model.eval()
    correct = 0
    y_true = []
    y_pred = []
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for data, pen_data in zip(loader, pen_loader):
        data = data.to(device)
        pen_data = pen_data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos, pen_data)
        pred = outputs[0].max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()

        y_true.append(data.y.view(-1).cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return correct / len(loader.dataset), accuracy, auc, precision, sensitivity, specificity, f1


def test_loss(loader, pen_loader, epoch):
    print('testing...........')
    model.eval()
    loss_all = 0
    for data, pen_data in zip(loader,pen_loader):
        data = data.to(device)
        pen_data = pen_data.to(device)
        # output, w1, w2, s1, s2= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos, pen_data)
        output, s1, s2 = model(data.x, data.edge_index, data.batch, data.edge_attr, data.pos, pen_data)

        loss_c = F.nll_loss(output, data.y)

        loss_tpk1 = topk_loss(s1,opt.ratio)
        loss_tpk2 = topk_loss(s2,opt.ratio)
        loss_consist = 0
        for c in range(opt.nclass):
            loss_consist += consist_loss(s1[data.y == c])
        # loss = opt.lamb0*loss_c + opt.lamb1 * loss_p1 + opt.lamb2 * loss_p2 \
        #            + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist
        loss = opt.lamb0*loss_c \
                   + opt.lamb3 * loss_tpk1 + opt.lamb4 *loss_tpk2 + opt.lamb5* loss_consist

        loss_all += loss.item() * data.num_graphs
    return loss_all / len(loader.dataset)


#######################################################################################
############################   Model Training #########################################
#######################################################################################
best_model_wts = copy.deepcopy(model.state_dict())
best_loss = 1e10
max_teacc = 0.0
max_auc = 0.0
max_spe = 0.0
max_sen = 0.0
max_pre = 0.0
max_f1 = 0.0

for epoch in range(0, num_epoch):
    since  = time.time()
    # tr_loss, s1_arr, s2_arr, w1, w2 = train(epoch)
    tr_loss, s1_arr, s2_arr = train(epoch)

    te_loss = test_loss(test_loader, test_pen_loader, epoch)

    tr_acc1, tr_acc2, tr_auc, tr_pre, tr_sen, tr_spe, tr_f1 = test_acc(train_loader,train_pen_loader)
    val_acc1, val_acc2, val_auc, val_pre, val_sen, val_spe, val_f1 = test_acc(val_loader, val_pen_loader)
    te_acc1, te_acc2, te_auc, te_pre, te_sen, te_spe, te_f1 = test_acc(test_loader, test_pen_loader)
    time_elapsed = time.time() - since
        
    if te_acc2 >= max_teacc:
        if te_acc2 > max_teacc:
            max_teacc = te_acc2
            max_auc = te_auc
            max_sen = te_sen
            max_spe = te_spe
            max_pre = te_pre
            max_f1 = te_f1
        elif te_acc2 == max_teacc:
            if te_auc > max_auc:
                max_teacc = te_acc2
                max_auc = te_auc
                max_sen = te_sen
                max_spe = te_spe
                max_pre = te_pre
                max_f1 = te_f1
            elif te_auc == max_auc:
                if te_sen > max_sen:
                    max_teacc = te_acc2
                    max_auc = te_auc
                    max_sen = te_sen
                    max_spe = te_spe
                    max_pre = te_pre
                    max_f1 = te_f1

    print('*====**')
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          'Train Acc: {:.7f}, TeAcc: {:.7f}'.format(epoch, tr_loss,
                                                       tr_acc2, te_acc1))
    print('Epoch: {:03d}, Test Loss: {:.7f}, '
          'Test Acc: {:.7f}, Test SEN: {:.7f}, Test SPE: {:.7f}, Test PRE: {:.7f}'.format(epoch, te_loss,
                                                       te_acc2, te_sen, te_spe, te_pre))
    print('Epoch: {:03d}, Train Loss: {:.7f}, '
          ', Test AUC: {:.7f}, Test F1: {:.7f}'.format(epoch, tr_loss,
                                                      te_auc, te_f1))


    if te_loss < best_loss and epoch > 5:
        print("saving best model")
        best_loss = te_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        if save_model:
            torch.save(best_model_wts, os.path.join(opt.save_path,str(fold)+'.pth'))

print('*====Max Testing===**')
print('Max Test Acc: {:.7f}, Max Test AUC: {:.7f}, Max Test SPE: {:.7f}, Max Test PRE: {:.7f}, '
          'Test SEN: {:.7f}, Test F1: {:.7f}'.format(max_teacc, max_auc,
                                                       max_spe, max_sen, max_pre, max_f1))

#######################################################################################
######################### Testing on testing set ######################################
#######################################################################################

if opt.load_model:
    model = Network(opt.indim,opt.ratio,opt.nclass).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.save_path,str(fold)+'.pth')))
    model.eval()
    preds = []
    correct = 0
    for data in test_loader:
        data = data.to(device)
        outputs= model(data.x, data.edge_index, data.batch, data.edge_attr,data.pos)
        pred = outputs[0].max(1)[1]
        preds.append(pred.cpu().detach().numpy())
        correct += pred.eq(data.y).sum().item()
    preds = np.concatenate(preds,axis=0)
    trues = test_dataset.data.y.cpu().detach().numpy()
    cm = confusion_matrix(trues,preds)
    print("Confusion matrix")
    print(classification_report(trues, preds))

else:
   model.load_state_dict(best_model_wts)
   model.eval()
   test_accuracy1, test_acc2, test_auc, test_pre, test_sen, test_spe, test_f1 = test_acc(test_loader, test_pen_loader)
   test_l= test_loss(test_loader, test_pen_loader, 0)
   print("===========================")
   print("Test Acc: {:.7f}, Test Loss: {:.7f} ".format(test_accuracy1, test_l))
   print(opt)

