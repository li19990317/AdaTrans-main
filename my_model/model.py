import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from my_model.ban import BANLayer
from my_model.fc import FCN
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=3,
                     stride=stride, padding=1, bias=False)
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.elu(out)
        return out



class DrugVQA(torch.nn.Module):
    """
    The class is an implementation of the DrugVQA my_model including regularization and without pruning.
    Slight modifications have been done for speedup
    
    """
    def __init__(self,args,block):
        """
        Initializes parameters suggested in paper
 
        args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            n_chars_smi : {int} voc size of smiles
            n_chars_seq : {int} voc size of protein sequence
            dropout     : {float}
            in_channels : {int} channels of CNN block input
            cnn_channels: {int} channels of CNN block
            cnn_layers  : {int} num of layers of each CNN block
            emb_dim     : {int} embeddings dimension
            dense_hid   : {int} hidden dim for the output dense
            task_type   : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes
 
        Returns:
            self
        """
        super(DrugVQA,self).__init__()

        self.batch_size = args['batch_size']
        self.lstm_hid_dim = args['lstm_hid_dim']
        self.r = args['r']
        self.type = args['task_type']
        self.in_channels = args['in_channels']
        #rnn
        self.embeddings = nn.Embedding(args['n_chars_smi'], args['emb_dim'])
        self.seq_embed = nn.Embedding(args['n_chars_seq'],args['emb_dim'])

        self.lstm = torch.nn.LSTM(args['emb_dim'],self.lstm_hid_dim,2,batch_first=True,bidirectional=True,dropout=args['dropout']) 
        self.linear_first = torch.nn.Linear(2*self.lstm_hid_dim,args['d_a'])
        self.linear_second = torch.nn.Linear(args['d_a'],args['r'])
        self.linear_first_seq = torch.nn.Linear(args['cnn_channels'],args['d_a'])
        self.linear_second_seq = torch.nn.Linear(args['d_a'],self.r)
        embedding_dim=args['embedding_dim']
        num_filters=args['num_filters']
        kernel_size=args['kernel_size']
        padding=args['padding']
        #CNN
        if padding:
            self.embedding = nn.Embedding(26, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(26, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size
        self.conv1 = nn.Conv1d(in_channels=in_ch[0], out_channels=in_ch[1], kernel_size=kernels[0])
        self.bn1 = nn.BatchNorm1d(in_ch[1])
        self.conv2 = nn.Conv1d(in_channels=in_ch[1], out_channels=in_ch[2], kernel_size=kernels[1])
        self.bn2 = nn.BatchNorm1d(in_ch[2])
        self.conv3 = nn.Conv1d(in_channels=in_ch[2], out_channels=in_ch[3], kernel_size=kernels[2])
        self.bn3 = nn.BatchNorm1d(in_ch[3])

        self.linear_final_step = torch.nn.Linear(self.lstm_hid_dim * 2 + args['d_a'], args['dense_hid'])
        self.linear_final = torch.nn.Linear(args['dense_hid'], args['n_classes'])

        #ban
        self.ban = BANLayer(v_dim=128, q_dim=32, h_dim=160, h_out=2)
        self.out_a = FCN(input_dim=224, output_dim=1024)
        
    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied

        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
    #LSTM
    def init_hidden(self):


        return (Variable(torch.zeros(4,self.batch_size,self.lstm_hid_dim).cuda()),Variable(torch.zeros(4,self.batch_size,self.lstm_hid_dim)).cuda())



        smile_embed = self.embeddings(x1)         
        outputs, self.hidden_state = self.lstm(smile_embed,self.hidden_state)   #LSTM
        sentence_att = F.tanh(self.linear_first(outputs))       
        sentence_att = self.linear_second(sentence_att)       
        sentence_att = self.softmax(sentence_att,1)       
        sentence_att = sentence_att.transpose(1,2)
        sentence_embed = sentence_att@outputs
        avg_sentence_embed = torch.sum(sentence_embed,1)/self.r  #multi head


        v = self.embedding(x2.long())
        v = v.transpose(2, 1)
        v = self.bn1(F.relu(self.conv1(v)))
        v = self.bn2(F.relu(self.conv2(v)))
        v = self.bn3(F.relu(self.conv3(v)))
        v = v.view(v.size(0), v.size(2), -1)

        v = v[:, :, :32]
        v= v.view(1, 1185, 32)


        seq_att = F.tanh(self.linear_first_seq(v))
        seq_att = self.linear_second_seq(seq_att)
        seq_att = self.softmax(seq_att, 1)
        seq_att = seq_att.transpose(1, 2)
        seq_embed = seq_att @ v
        avg_seq_embed = torch.sum(v,1)/self.r
        interactin = self.ban(sentence_embed,seq_embed)
        sscomplex = torch.cat([avg_sentence_embed,avg_seq_embed],dim=1)
        sscomplex = F.relu(self.linear_final_step(sscomplex))
        interactin_2=torch.cat([interactin[0],sscomplex],dim=1)
        # binding_model = self.out_a()
        binding_affinity_pred = self.out_a(interactin_2)
        if not bool(self.type):
            output = F.sigmoid(self.linear_final(sscomplex))
            return output, seq_att, binding_affinity_pred
        else:
            return F.log_softmax(self.linear_final(sscomplex)),v,binding_affinity_pred.item()

