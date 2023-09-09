from C2AE import C2AE, save_model, load_model, Fe, Fx, Fd, eval_metrics
from handle_data_data import SpanClDataset
from torch.utils.data import DataLoader
from multilabel_loss import MultiLabelCircleLoss
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "1, 2"
 
from transformers import AdamW
from tqdm import tqdm
# from transformers import AdamW,WarmupLinearSchedule
# from tqdm import tqdm
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_predict(logits):
    output = []
    for i in logits:
        li = [0 for _ in range(14)]
        for j in range(14):
            if i[j] > 0:
                li[j] = 1
        output.append(li)
    return torch.tensor(output).to(device)
                
 
 
def train(model,train_loader,dev_loader):
    model.to(device)
    model.train()
    criterion = MultiLabelCircleLoss()
 
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    #设置模型参数的权重衰减
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    #学习率的设置
    optimizer_params = {'lr': 1e-5, 'eps': 1e-6, 'correct_bias': False}
    #AdamW 这个优化器是主流优化器
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_params)
 
    #学习率调整器，检测准确率的状态，然后衰减学习率
    scheduler = ReduceLROnPlateau(optimizer,mode='max',factor=0.5,min_lr=1e-7, patience=5,verbose= True, threshold=0.0001, eps=1e-08)
 
    t_total = len(train_loader)
    total_epochs = 150
    bestAcc = 0
    correct = 0
    total = 0
    print('Training begin!')
    for epoch in range(total_epochs):
        for step, (indextokens_a,input_mask_a,indextokens_b,input_mask_b,label) in enumerate(train_loader):
            indextokens_a,input_mask_a,indextokens_b,input_mask_b,label = indextokens_a.to(device),input_mask_a.to(device),indextokens_b.to(device),input_mask_b.to(device),label.to(device)
            optimizer.zero_grad()
            fx_x, fe_y, fd_z = model(indextokens_a,input_mask_a,indextokens_b,input_mask_b)
            c_loss,l_loss = model.losses(fx_x, fe_y, fd_z, label)
            c_loss = criterion(fd_z, label)
            # print("loss",c_loss)
            loss = model.beta*l_loss + model.alpha*c_loss

            # _, predict = torch.max(fd_z.data, 1)
            predict = get_predict(fd_z)
            for i in range(label.size(0)):
                check = True
                for j in range(14):
                    if predict[i][j] != label[i][j]:
                        check = False
                        break
                if check:
                    correct += 1
            # correct += (predict == label).sum().item()
            total += label.size(0)
            # print(fd_z.shape)
            # print(fd_z[0])
            loss.backward()
            optimizer.step()
            # print("Train Epoch[{}/{}],step[{}/{}], %,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),loss.item()))
            # if (step+1) % 50 == 0:
            #     path = 'model.pkl'
            #     torch.save(model, path)
            if (step + 1) % 2 == 0:
                train_acc = correct / total
                print("Train Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),train_acc*100,loss.item()))
 
            if (step + 1) % 20 == 0:
                train_acc = correct / total
                acc = dev(model, dev_loader)
                model.train()
                if bestAcc < acc:
                    bestAcc = acc
                    # path = 'model_Albert_nl2query_section2_albert.pkl'
                    # torch.save(model, path)
                print("DEV Epoch[{}/{}],step[{}/{}],tra_acc{:.6f} %,bestAcc{:.6f}%,dev_acc{:.6f} %,loss:{:.6f}".format(epoch + 1, total_epochs, step + 1, len(train_loader),train_acc*100,bestAcc*100,acc*100,loss.item()))
        scheduler.step(bestAcc)
 
def dev(model,dev_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for step, (
                indextokens_a, input_mask_a, indextokens_b, input_mask_b, label) in tqdm(enumerate(
            dev_loader),desc='Dev Itreation:'):
            indextokens_a, input_mask_a, indextokens_b, input_mask_b, label = indextokens_a.to(device), input_mask_a.to(
                device), indextokens_b.to(device), input_mask_b.to(device), label.to(device)
            fd_z = model(indextokens_a,input_mask_a,indextokens_b,input_mask_b)
            predict = get_predict(fd_z)
            for i in range(label.size(0)):
                check = True
                for j in range(14):
                    if predict[i][j] != label[i][j]:
                        check = False
                        break
                if check:
                    correct += 1
            # correct += (predict == label).sum().item()
            total += label.size(0)
            # total += label.size(0)
        res = correct / total
        return res
 
 
if __name__ == '__main__':
    batch_size = 48
    train_data = SpanClDataset('Groundtruth.txt')
    dev_data = SpanClDataset('Groundtruth.txt')
    # test_data = SpanClDataset('data/LCQMC/test.tsv')
 
 
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_data, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    feat_dim = 768
    num_labels = 768
    latent_dim = 50
    fx_h_dim=110
    fe_h_dim=105
    fd_h_dim=105

    Fx_tmc = Fx(feat_dim, fx_h_dim, fx_h_dim, latent_dim)
    Fe_tmc = Fe(num_labels, fe_h_dim, latent_dim)
    Fd_tmc = Fd(latent_dim, fd_h_dim, 14)
    # model = torch.load('model_Albert_nl2query_section2_albert.pkl')
    # model = torch.load('Career_Platform_Reforged/classifier/model/model_Albert.pkl')
    model = C2AE(Fx_tmc, Fe_tmc, Fd_tmc, beta= 0.001, alpha=65, emb_lambda=0.01, latent_dim=latent_dim, device=device)
    train(model,train_loader,dev_loader)

 
 
 
 
 
 
 
 
 
 
 
 