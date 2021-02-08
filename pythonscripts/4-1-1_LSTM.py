import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,accuracy_score,classification_report


dataset_1 = np.load('1.npy')
print('dataset_1 shape:', dataset_1.shape)

# check if a GPU is available
with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print('We are using %s now.' %device)

# remove instance with label==10 (rest)
removed_label = [2,3,4,5,6,7,8,9,10]  #2,3,4,5,
for ll in removed_label:
    id = dataset_1[:, -1]!=ll
    dataset_1 = dataset_1[id]

# data segmentation
n_class = int(11-len(removed_label))  # 0~9 classes ('10:rest' is not considered)
no_feature = 64  # the number of the features
segment_length = 16  # selected time window; 16=160*0.1
LR = 0.005  # learning rate
EPOCH = 101
n_hidden = 128  # number of neurons in hidden layer
l2 = 0.001  # the coefficient of l2-norm regularization

def one_hot(y_):
    # Function to encode output labels from number indexes
    # e.g.: [[5], [0], [3]] --> [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    y_ = y_.reshape(len(y_))
    y_ = [int(xx) for xx in y_]
    n_values = np.max(y_) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]

def extract(input, n_classes, n_fea, time_window, moving):
    xx = input[:, :n_fea]
    yy = input[:, n_fea:n_fea + 1]
    new_x = []
    new_y = []
    number = int((xx.shape[0] / moving) - 1)
    for i in range(number):
        ave_y = np.average(yy[int(i * moving):int(i * moving + time_window)])
        if ave_y in range(n_classes + 1):
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(ave_y)
        else:
            new_x.append(xx[int(i * moving):int(i * moving + time_window), :])
            new_y.append(0)

    new_x = np.array(new_x)
    new_x = new_x.reshape([-1, n_fea * time_window])
    new_y = np.array(new_y)
    new_y.shape = [new_y.shape[0], 1]
    data = np.hstack((new_x, new_y))
    data = np.vstack((data, data[-1]))  # add the last sample again, to make the sample number round
    return data

data_seg = extract(dataset_1, n_classes=n_class, n_fea=no_feature, time_window=segment_length, moving=(segment_length/2))  # 50% overlapping
print('After segmentation, the shape of the data:', data_seg.shape)

# split training and test data
no_longfeature = no_feature*segment_length
data_seg_feature = data_seg[:, :no_longfeature]
data_seg_label = data_seg[:, no_longfeature:no_longfeature+1]
train_feature, test_feature, train_label, test_label = train_test_split(data_seg_feature, data_seg_label,test_size=0.2, shuffle=True)

# normalization
# before normalize reshape data back to raw data shape
train_feature_2d = train_feature.reshape([-1, no_feature])
test_feature_2d = test_feature.reshape([-1, no_feature])

scaler1 = StandardScaler().fit(train_feature_2d)
train_fea_norm1 = scaler1.transform(train_feature_2d) # normalize the training data
test_fea_norm1 = scaler1.transform(test_feature_2d) # normalize the test data
print('After normalization, the shape of training feature:', train_fea_norm1.shape,
      '\nAfter normalization, the shape of test feature:', test_fea_norm1.shape)

# after normalization, reshape data to 3d in order to feed in to LSTM
train_fea_norm1 = train_fea_norm1.reshape([-1, segment_length, no_feature])
test_fea_norm1 = test_fea_norm1.reshape([-1, segment_length, no_feature])
print('After reshape, the shape of training feature:', train_fea_norm1.shape,
      '\nAfter reshape, the shape of test feature:', test_fea_norm1.shape)

BATCH_size = test_fea_norm1.shape[0] # use test_data as batch size

# feed data into dataloader
train_fea_norm1 = torch.tensor(train_fea_norm1).to(device)
train_label = torch.tensor(train_label.flatten()).to(device)
train_data = Data.TensorDataset(train_fea_norm1, train_label)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_size, shuffle=False)

test_fea_norm1 = torch.tensor(test_fea_norm1).to(device)
test_label = torch.tensor(test_label.flatten()).to(device)

# classifier
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm_layer = nn.LSTM(
            input_size=no_feature,
            hidden_size=n_hidden,         # LSTM hidden unit
            num_layers=2,           # number of LSTM layer
            bias=True,
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, segment_length, no_feature)
        )

        self.out = nn.Linear(n_hidden, n_class)

    def forward(self, x):
        r_out, (h_n, h_c) = self.lstm_layer(x.float(), None)
        r_out = F.dropout(r_out, 0.3)

        test_output = self.out(r_out[:, -1, :]) # choose r_out at the last time step
        return test_output

lstm = LSTM()
lstm.to(device)
print(lstm)

optimizer = torch.optim.Adam(lstm.parameters(), lr=LR, weight_decay=l2)   # optimize all parameters
loss_func = nn.CrossEntropyLoss()

best_acc = []
best_auc = []

# training and testing
start_time = time.perf_counter()
for epoch in range(EPOCH):
    for step, (train_x, train_y) in enumerate(train_loader):

        output = lstm(train_x)  # LSTM output of training data
        loss = loss_func(output, train_y.long())  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    if epoch % 10 == 0:
        test_output = lstm(test_fea_norm1)  # LSTM output of test data
        test_loss = loss_func(test_output, test_label.long())

        test_y_score = one_hot(test_label.data.cpu().numpy())  # .cpu() can be removed if your device is cpu.
        pred_score = F.softmax(test_output, dim=1).data.cpu().numpy()  # normalize the output
        auc_score = roc_auc_score(test_y_score, pred_score)

        pred_y = torch.max(test_output, 1)[1].data.cpu().numpy()
        pred_train = torch.max(output, 1)[1].data.cpu().numpy()

        test_acc = accuracy_score(test_label.data.cpu().numpy(), pred_y)
        train_acc = accuracy_score(train_y.data.cpu().numpy(), pred_train)


        print('Epoch: ', epoch, '|train loss: %.4f' % loss.item(),
              ' train ACC: %.4f' % train_acc, '| test loss: %.4f' % test_loss.item(),
              'test ACC: %.4f' % test_acc, '| AUC: %.4f' % auc_score)
        best_acc.append(test_acc)
        best_auc.append(auc_score)

current_time = time.perf_counter()
running_time = current_time - start_time
print(classification_report(test_label.data.cpu().numpy(), pred_y))
print('BEST TEST ACC: {}, AUC: {}'.format(max(best_acc), max(best_auc)))
print("Total Running Time: {} seconds".format(round(running_time, 2)))
