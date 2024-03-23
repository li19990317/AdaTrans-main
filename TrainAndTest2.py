from utils import *
from ArgsPre import *
from sklearn import metrics
import sys
import pandas as pd
import time
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def train(trainArgs, dataLoader = None):

    """
    args:
        my_model           : {object} my_model
        lr              : {float} learning rate
        train_loader    : {DataLoader} training data loaded into a dataloader
        doTest          : {bool} do test_module or not
        test_proteins   : {list} proteins list for test_module
        testDataDict    : {dict} test_module data dict
        seqContactDict  : {dict} seq-contact dict
        optimizer       : optimizer
        criterion       : loss function. Must be BCELoss for binary_classification and NLLLoss for multiclass
        epochs          : {int} number of epochs
        use_regularizer : {bool} use penalization or not
        penal_coeff     : {int} penalization coeff
        clip            : {bool} use gradient clipping or not
    Returns:
        accuracy and losses of the my_model
    """
    losses = []
    accs = []
    A_Loss= nn.MSELoss()
    for i in range(trainArgs['epochs']):
        print("Running EPOCH",i+1)
        total_loss = 0
        n_batches = 0
        correct = 0
        train_loader = dataLoader
        optimizer = trainArgs['optimizer']
        criterion = trainArgs["criterion"]
        attention_model = trainArgs['my_model']
        for batch_idx, (lines, seq, label,properties) in enumerate(train_loader):
            input, seq_lengths, y = make_variables(lines, properties,smiles_letters)
            attention_model.hidden_state = attention_model.init_hidden()
            seq = create_variable(seq)
            y_pred,att, binding_affinity_pred = attention_model(input,seq)
            label = label.float()
            binding_affinity_pred = binding_affinity_pred.float()
            label = label.to(device)
            binding_affinity_pred = binding_affinity_pred.to(device)


            #penalization AAT - I
            if trainArgs['use_regularizer']:
                attT = att.transpose(1,2)
                identity = torch.eye(att.size(1))
                identity = Variable(identity.unsqueeze(0).expand(train_loader.batch_size,att.size(1),att.size(1))).cuda()
                penal = attention_model.l2_matrix_norm(att@attT - identity)

            if not bool(attention_model.type) :
                #print(bool(attention_model.type)) =False
                #binary classification
                #Adding a very small value to prevent BCELoss from outputting NaN's
                correct+=torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),y.type(torch.DoubleTensor)).data.sum()
                if trainArgs['use_regularizer']:
                    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))+(trainArgs['penal_coeff'] * penal.cpu()/train_loader.batch_size)

                else:
                    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))
                    loss2 = A_Loss(label, binding_affinity_pred)


            total_loss = loss + loss2
            Total_loss = total_loss.data

            optimizer.zero_grad()
            total_loss.backward()#retain_graph=True



            # gradient clipping
            if trainArgs['clip']:
                torch.nn.utils.clip_grad_norm(attention_model.parameters(), 0.5)
            optimizer.step()
            n_batches += 1
            if batch_idx % 1000 == 0:
                print(batch_idx)

        avg_loss = Total_loss / n_batches
        acc = correct.numpy() / (len(train_loader.dataset))

        losses.append(avg_loss)
        accs.append(acc)



        with open('train_logs.txt', 'w') as f:
            sys.stdout = f


            print("Training process:")
            print("avg_loss is", avg_loss)
            print("train ACC = ", acc)

            sys.stdout = sys.__stdout__



        if(trainArgs['doTest']):
            testArgs = {}
            testArgs['my_model'] = trainArgs['my_model']
            testArgs['criterion'] = trainArgs['criterion']
            testArgs['use_regularizer'] = trainArgs['use_regularizer']
            testArgs['penal_coeff'] = trainArgs['penal_coeff']
            testArgs['clip'] = trainArgs['clip']


    return losses,accs
def getROCE(predList,targetList,roceRate):
    p = sum(targetList)
    n = len(targetList) - p
    predList = [[index,x] for index,x in enumerate(predList)]
    predList = sorted(predList,key = lambda x:x[1],reverse = True)
    tp1 = 0
    fp1 = 0
    maxIndexs = []
    for x in predList:
        if(targetList[x[0]] == 1):
            tp1 += 1
        else:
            fp1 += 1
            if(fp1>((roceRate*n)/100)):
                break
    roce = (tp1*n)/(p*fp1)
    return roce

def test(testArgs,dataLoader = None):
    test_loader = dataLoader
    testArgs['test_loader'] = test_loader
    test_loader = testArgs['test_loader']
    print("test loader....")
    criterion = testArgs["criterion"]
    attention_model = testArgs['my_model']
    A_Loss = nn.MSELoss()
    losses = []
    accuracy = []
    print('test_module begin ...')
    total_loss = 0
    n_batches = 0
    correct = 0
    all_pred = np.array([])
    all_target = np.array([])
    true_labels_list = np.array([])
    pred_labels_list = np.array([])
    with torch.no_grad():
        for batch_idx,(lines,seq,label,properties) in enumerate(test_loader):
            input, seq_lengths, y = make_variables(lines, properties,smiles_letters)
            attention_model.hidden_state = attention_model.init_hidden()
            seq = create_variable(seq)
            y_pred,att, binding_affinity_pred = attention_model(input,seq)
            label = label.float()
            binding_affinity_pred = binding_affinity_pred.float()
            label = label.to(device)
            binding_affinity_pred = binding_affinity_pred.to(device)


            if not bool(attention_model.type) :
                #binary classification
                #Adding a very small value to prevent BCELoss from outputting NaN's
                pred = torch.round(y_pred.type(torch.DoubleTensor).squeeze(1))
                correct+=torch.eq(torch.round(y_pred.type(torch.DoubleTensor).squeeze(1)),y.type(torch.DoubleTensor)).data.sum()
                all_pred=np.concatenate((all_pred,y_pred.data.cpu().squeeze(1).numpy()),axis = 0)
                all_target = np.concatenate((all_target,y.data.cpu().numpy()),axis = 0)
                pred_labels_list = np.concatenate((pred_labels_list, binding_affinity_pred.data.cpu().squeeze(1).numpy()), axis=0)
                true_labels_list = np.concatenate((true_labels_list, label.data.cpu().numpy()), axis=0)
                alpha = 0.4
                if trainArgs['use_regularizer']:
                    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))# (C * penal.cpu()/train_loader.batch_size)
                else:
                    loss = criterion(y_pred.type(torch.DoubleTensor).squeeze(1),y.type(torch.DoubleTensor))
                    loss2 = A_Loss(torch.from_numpy(pred_labels_list), torch.from_numpy(true_labels_list)).item()
            total_loss+=loss.data
            Total_loss = alpha *loss +(1-alpha)* loss2
            n_batches+=1
    with open('test_logs.txt', 'w') as f:
        sys.stdout = f
        CI = get_cindex(true_labels_list, pred_labels_list)
        MSE = mean_squared_error(true_labels_list, pred_labels_list)
        rm = get_rm2(true_labels_list, pred_labels_list)
        testSize = round(len(test_loader.dataset),3)
        testAcc = round(correct.numpy()/(n_batches*test_loader.batch_size),3)
        testRecall = round(metrics.recall_score(all_target,np.round(all_pred)),3)
        testPrecision = round(metrics.precision_score(all_target,np.round(all_pred)),3)
        testAuc = round(metrics.roc_auc_score(all_target, all_pred),3)
        AUPR_i= metrics.average_precision_score(all_target, all_pred)
        AUPR_a = get_aupr(true_labels_list, pred_labels_list)
        print("AUPR_i = ", AUPR_i)
        print("AUPR_a = ", AUPR_a)
        print("CI =" , CI)
        print("MSE =" , MSE)
        print("rm =" , rm)

        testLoss = round(Total_loss.item()/n_batches,5)
        print("test_module size =",testSize,"  test_module acc =",testAcc,"  test_module recall =",testRecall,"  test_module precision =",testPrecision,"  test_module auc =",testAuc,"  test_module loss = ",testLoss)
        roce1 = round(getROCE(all_pred,all_target,0.5),2)
        roce2 = round(getROCE(all_pred,all_target,1),2)
        roce3 = round(getROCE(all_pred,all_target,2),2)
        roce4 = round(getROCE(all_pred,all_target,5),2)
        print("roce0.5 =",roce1,"  roce1.0 =",roce2,"  roce2.0 =",roce3,"  roce5.0 =",roce4)
        sys.stdout = sys.__stdout__
        return testAcc,testRecall,testPrecision,testAuc,testLoss,all_pred,all_target,pred_labels_list,true_labels_list,roce1,roce2,roce3,roce4,CI,MSE,rm, AUPR_i,AUPR_a
def train_test(trainArgs):
    log_directory = './logs/'
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    nowtime = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())



    train_sets =['./data/Davis/davis3/CV1_DAVIS_unseenD_useenP_train.txt',
         './data/Davis/davis3/CV2_DAVIS_unseenD_useenP_train.txt',
         './data/Davis/davis3/CV3_DAVIS_unseenD_useenP_train.txt',
         './data/Davis/davis3/CV4_DAVIS_unseenD_useenP_train.txt',
         './data/Davis/davis3/CV5_DAVIS_unseenD_useenP_train.txt']
    valid_sets =['./data/Davis/davis3/CV1_DAVIS_unseenD_useenP_val.txt',
         './data/Davis/davis3/CV2_DAVIS_unseenD_useenP_val.txt',
         './data/Davis/davis3/CV3_DAVIS_unseenD_useenP_val.txt',
         './data/Davis/davis3/CV4_DAVIS_unseenD_useenP_val.txt',
         './data/Davis/davis3/CV5_DAVIS_unseenD_useenP_val.txt']

    for fold_idx, (train_set, valid_set) in enumerate(zip(train_sets, valid_sets), 1):
        # print(train_set)
        trainDataSet = getTrainDataSet(train_set)
        validDataSet = getTestDataSet(valid_set)
        trainset = ProDataset(dataSet =trainDataSet)
        validset = ProDataset(dataSet =validDataSet)
        train_loader1 = DataLoader(dataset=trainset, batch_size=1, shuffle=True)
        valid_loader1 = DataLoader(dataset=validset, batch_size=1, shuffle=True)
        # print(fold_idx)
        losses, accs = train(trainArgs,dataLoader = train_loader1)
        testAcc, testRecall, testPrecision, testAuc, testLoss, all_pred, all_target,pred_labels_list,true_labels_list, roce1, roce2, roce3, roce4, CI, MSE, rm, AUPR_i,AUPR_a= test(testArgs, dataLoader = valid_loader1)

        print_log(fold_idx,losses,accs,testAcc, testRecall, testPrecision, testAuc, testLoss, roce1, roce2, roce3, roce4, CI, MSE, rm,AUPR_i,AUPR_a,nowtime)


if __name__ == '__main__':
    losses = 0
    testArgs = {}
    testArgs['my_model'] = trainArgs['my_model']
    testArgs['criterion'] = trainArgs['criterion']
    testArgs['use_regularizer'] = trainArgs['use_regularizer']
    testArgs['penal_coeff'] = trainArgs['penal_coeff']
    testArgs['clip'] = trainArgs['clip']
    train_test(trainArgs)
