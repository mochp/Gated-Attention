import time
import torch.optim as optim
import torch
import random
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torchtext import data, datasets
from transformers import BertTokenizer, BertModel


N_EPOCHS = 5
N_CIRCLE = 5
BATCH_SIZE = 8
CUT_LENGTH = 8
MAX_LENGTH = 128
HIDDEN_DIM = 256
OUTPUT_DIM = 2
N_LAYERS = 1
BIDIRECTIONAL = True
DROPOUT = 0.25

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer.tokenize('Hello WORLD how ARE yoU?')
indexes = tokenizer.convert_tokens_to_ids(tokens)

max_input_length = tokenizer.max_model_input_sizes['bert-base-uncased']


def cut(key,max_len):
    lists=[]
    num=len(key)
    for i in range(num-1):
        A=[0 for x in range(0,max_len)]
        B=[1 for x in range(key[i],key[i+1])]
        A[key[i]:key[i+1]]=B
        lists.append(A)
    length=len(lists)
    if length<8:
        for i in range(length+1,9):
            A = [0 for x in range(0, max_len)]
            lists.append(A)
    return lists



def gate_token(gate):
    gate = gate.split(" ")
    gate = [int(x) for x in gate]
    mask = cut(gate,128)
    return mask


def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:MAX_LENGTH-2]
    if len(tokens) < MAX_LENGTH-2:
        for i in range(MAX_LENGTH-2-len(tokens)):
            tokens.append(tokenizer.pad_token)

    return tokens


TEXT = data.Field(batch_first=True,
                  use_vocab=False,
                  fix_length=MAX_LENGTH,
                  tokenize=tokenize_and_cut,
                  preprocessing=tokenizer.convert_tokens_to_ids,
                  init_token=tokenizer.cls_token_id,
                  eos_token=tokenizer.sep_token_id,
                  pad_token=tokenizer.pad_token_id,
                  unk_token=tokenizer.unk_token_id)

LABEL = data.LabelField(dtype=torch.float)

GATE = data.Field(sequential=True,
                  batch_first=True,
                  use_vocab=False,
                  tokenize=gate_token)


fields = [('label', LABEL), ('text', TEXT), ('gate', GATE)]

train_data, valid_data, test_data = data.TabularDataset.splits(
    path='dataset/stsa_seg_bert',
    train='train.csv',
    validation='dev.csv',
    test='test.csv',
    format='csv',
    fields=fields,
    skip_header=True)

print(f"Number of training examples: {len(train_data)}")
print(f"Number of validation examples: {len(valid_data)}")
print(f"Number of testing examples: {len(test_data)}")

print(vars(train_data.examples[6]))


tokens = tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])
print(tokens)


LABEL.build_vocab(train_data)
print(LABEL.vocab.stoi)


train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    sort=False,
    batch_size=BATCH_SIZE,
    device=device)


class Attention(torch.nn.Module):

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.zeros(hidden_dim * 2))

    def forward(self, x):
        H0 = self.tanh(x)
        H1 = torch.matmul(H0, self.w)
        H2 = nn.functional.softmax(H1, dim=1)
        alpha = H2.unsqueeze(-1)
        att_hidden = torch.sum(x * alpha, 1)
        return att_hidden, H2


class Entropy(torch.nn.Module):

    def __init__(self):
        super(Entropy, self).__init__()
        self.linear = nn.Linear(1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        log_x = torch.log(x)
        v = -1*torch.sum(x*log_x, axis=-1)
        v = v.unsqueeze(-1)
        m = self.linear(v)
        e = self.sigmoid(m)
        return e


class AuxiliaryNet(torch.nn.Module):

    def __init__(self, hidden_dim, cut_length, max_length, tau=1):
        super(AuxiliaryNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.cut_length = cut_length
        self.max_length = max_length
        self.linear1 = nn.Linear(self.hidden_dim * 2, cut_length)
        self.linear2 = nn.Linear(self.max_length, 1)
        self.sigmoid = nn.Sigmoid()
        self.tau = tau

    def forward(self, lstm_hidden, alpha, mask):
        # step1: output 8 gate
        linear1 = self.linear1(lstm_hidden)
        reshape = linear1.view([-1, self.cut_length, self.max_length])
        linear2 = self.linear2(reshape)
        p_t = self.sigmoid(linear2)
        p_t = p_t.repeat(1, 1, 2)
        p_t[:, :, 0] = 1 - p_t[:, :, 0]
        g_hat = nn.functional.gumbel_softmax(p_t, self.tau, hard=False)
        g_t = g_hat[:, :, 1]  #[8,8] 

        # step2: attention and  gate and mask
        alpha_permute = alpha.unsqueeze(-1).permute([0,2,1])  #[8,1,128]
        # mask_view = mask.view([-1, self.cut_length, self.max_length])    #[8,8,128] 

        g_t_mask = torch.matmul(g_t, mask*1.0)  #[b,cut,len] 
        g_t_mask_attention = g_t_mask*alpha_permute  #[8,8,128] 
        # step3: and hidden
        output_hidden = torch.bmm(g_t_mask_attention,lstm_hidden)
        sum_hidden = torch.sum(output_hidden,axis=1) 
        #step4: mean
        mask_sum = torch.sum(mask,axis=2)
        mask_sum1 = torch.gt(mask_sum,1)
        mask_sum2 = torch.sum(mask_sum1,axis=1)
        sum_hidden = torch.div(sum_hidden,mask_sum2.unsqueeze(-1))

        return sum_hidden


class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 cut_length,
                 max_length,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):

        super().__init__()
        self.bert = bert
        self.cut_length = cut_length
        self.max_length = max_length
        embedding_dim = bert.config.to_dict()['hidden_size']

        self.entropy = Entropy()
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers=n_layers,
                          bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.auxiliary = AuxiliaryNet(hidden_dim, cut_length, max_length)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, text, mask):
        with torch.no_grad():
            embedded = self.bert(text)[0]
        rnn_hidden, _ = self.rnn(embedded)
        # att_hidden
        att_hidden, alpha = self.attention(rnn_hidden)
        # gate_hidden
        gate_hidden = self.auxiliary(rnn_hidden, alpha, mask)
        # att_hidden and gate_hidden
        v = self.entropy(alpha)
        sum_hidden = att_hidden+(1-v)*gate_hidden
        hidden = self.fc(sum_hidden)
        hidden = self.softmax(hidden)
        return hidden


model = BERTGRUSentiment(BertModel.from_pretrained('bert-base-uncased'),
                         CUT_LENGTH,
                         MAX_LENGTH,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

# In order to freeze paramers (not train them) we need to set their `requires_grad` attribute to `False`. To do this, we simply loop through all of the `named_parameters` in our model and if they're a part of the `bert` transformer model, we set `requires_grad = False`.


for name, param in model.named_parameters():
    if name.startswith('bert'):
        param.requires_grad = False


# We can now see that our model has under 3M trainable parameters, making it almost comparable to the `FastText` model. However, the text still has to propagate through the transformer which causes training to take considerably longer.


print(f'The model has {count_parameters(model):,} trainable parameters')


# We can double check the names of the trainable parameters, ensuring they make sense. As we can see, they are all the parameters of the GRU (`rnn`) and the linear layer (`out`).


for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

# ## Train the Model

optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()


# Place the model and criterion onto the GPU (if available)


model = model.to(device)
criterion = criterion.to(device)


# Next, we'll define functions for: calculating accuracy, performing a training epoch, performing an evaluation epoch and calculating how long a training/evaluation epoch takes.


def binary_accuracy(mei, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    # round predictions to the closest integer
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()  # convert into float for division
    acc = correct.sum() / len(correct)
    return acc


def train(model, iterator, optimizer, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in tqdm(iterator):
        bayes_loss = []
        optimizer.zero_grad()

        # 循环50次，最大似然损失
        for i in range(N_CIRCLE):
            # 
            predictions = model(batch.text, batch.gate)
            bayes_loss.append(torch.gather(predictions, 1, batch.label.long().unsqueeze(-1)))

        possible = torch.cat(tuple([x for x in bayes_loss]),1)
        possible_max = torch.max(possible,dim=1).values

        loss = sum(1-possible_max) 
        acc_num = sum(torch.gt(possible_max,0.5))
        acc =  torch.floor_divide(acc_num,len(possible_max)) 

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator),epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(iterator):
            bayes_predict = []
            optimizer.zero_grad()
            # 循环50次，最大似然损失
            for i in range(N_CIRCLE):
                predictions = model(batch.text, batch.gate)
                bayes_predict.append(predictions)

            predictions = torch.cat(tuple([x for x in bayes_predict]),0)
            predictions1 = predictions.view((N_CIRCLE,BATCH_SIZE,2)).permute((1,0,2))
            left,right =  torch.chunk(predictions1, 2, dim=2)
            left_sub_right = abs(left-right)
            index = torch.max(left_sub_right,dim=1).indices
            chiose_left = torch.gather(predictions1, 1, index.unsqueeze(-1)).view((-1))
            chiose_right = 1-chiose_left
            result = torch.cat((chiose_left.unsqueeze(-1),chiose_right.unsqueeze(-1)),axis=1)

            loss += criterion(result, batch.label.long())

            acc_0 = torch.gather(result, 1, batch.label.long().unsqueeze(-1))
            acc_1 = sum(torch.gt(acc_0,0.5))
            acc =  torch.floor_divide(acc_1,len(acc_0)) 


            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator),epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# Finally, we'll train our model. This takes considerably longer than any of the previous models due to the size of the transformer. Even though we are not training any of the transformer's parameters we still need to pass the data through the model which takes a considerable amount of time on a standard GPU.


best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')

    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')


# We'll load up the parameters that gave us the best validation loss and try these on the test set - which gives us our best results so far!


model.load_state_dict(torch.load('tut6-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

