# from model import *
from utils import *
from torch.utils.data import Dataset, DataLoader
from my_model.model import *

class ProDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self,dataSet):
        self.dataSet = dataSet#list:[[smile,seq,label,letter],....]
        self.len = len(dataSet)

        self.properties_i = [int(x[3]) for x in dataSet]#letters
        self.properties = [float(x[2]) for x in dataSet]# labels
        self.property_list = list(sorted(set(self.properties)))

    def __getitem__(self, index):
        smiles,seq,label,letter = self.dataSet[index]
        # contactMap = one_hot_encode(seq)
        seq = integer_label_protein(seq)
        return smiles, seq, float(label), int(letter)

    def __len__(self):
        return self.len

    def get_properties(self):
        return self.property_list

    def get_property(self, id):
        return self.property_list[id]

    def get_property_id(self, property):
        return self.property_list.index(property)

def one_hot_encode(sequence):
    alphabet = sorted(set(sequence))
    char_to_index = {char: index for index, char in enumerate(alphabet)}
    encoding = torch.zeros(1,len(sequence), len(char_to_index))
    for i, aa in enumerate(sequence):
        if aa in char_to_index:
            index = char_to_index[aa]
            encoding[0][i][index] = 1
    return encoding



smileLettersPath  = './data/voc/combinedVoc-wholeFour.voc'
seqLettersPath = './data/voc/sequence.voc'
print('get train datas....')
print('get letters....')
smiles_letters = getLetters(smileLettersPath)
sequence_letters = getLetters(seqLettersPath)



print('get protein-seq dict....')

N_CHARS_SMI = len(smiles_letters)
N_CHARS_SEQ = len(sequence_letters)

print('train loader....')
print('my_model args...')

modelArgs = {}
modelArgs['batch_size'] = 1
modelArgs['lstm_hid_dim'] = 64
modelArgs['d_a'] = 32
modelArgs['r'] = 10
modelArgs['n_chars_smi'] = 496 
modelArgs['n_chars_seq'] = 21
modelArgs['dropout'] = 0.2
modelArgs['in_channels'] = 8
modelArgs['cnn_channels'] = 32
modelArgs['cnn_layers'] = 4
modelArgs['emb_dim'] = 30
modelArgs['dense_hid'] = 64
modelArgs['task_type'] = 0
modelArgs['n_classes'] = 1
modelArgs['embedding_dim'] =128
modelArgs['num_filters'] = [128,128,128]
modelArgs['kernel_size'] = [3,6,9]
modelArgs['padding'] = True
print('train args...')

trainArgs = {}
trainArgs['my_model'] = DrugVQA(modelArgs,block = ResidualBlock).cuda()
trainArgs['epochs'] = 30
trainArgs['lr'] = 0.0007

trainArgs['doTest'] = True

trainArgs['use_regularizer'] = False
trainArgs['penal_coeff'] = 0.03
trainArgs['clip'] = True
trainArgs['criterion'] = torch.nn.BCELoss()
trainArgs['optimizer'] = torch.optim.Adam(trainArgs['my_model'].parameters(),lr=trainArgs['lr'])


print('train args over...')
