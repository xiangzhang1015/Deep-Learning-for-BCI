import time
import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score,classification_report
from sklearn.preprocessing import StandardScaler


# np.random.seed(0)

dataset_1 = np.load('1.npy')
print('dataset_1 shape:', dataset_1.shape)

# check if a GPU is available
with_gpu = torch.cuda.is_available()
if with_gpu:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# remove instance with label==10 (rest)
removed_label = [2,3,4,5,6,7,8,9,10]  #2,3,4,5,
for ll in removed_label:
    id = dataset_1[:, -1]!=ll
    dataset_1 = dataset_1[id]

# data segmentation
n_class = int(11-len(removed_label))  # 0~9 classes ('10:rest' is not considered)
no_feature = 64  # the number of the features
segment_length = 16  # selected time window; 16=160*0.1
LR = 0.001  # learning rate
EPOCH = 101

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
data_seg_feature = data_seg[:, :1024]
data_seg_label = data_seg[:, 1024:1025]
train_feature, test_feature, train_label, test_label = train_test_split(data_seg_feature, data_seg_label, shuffle=True)

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

# Create adjacency matrix (edge index). Here we take complete graph as an example (all nodes are connected to each other)
# You may design your own edge index in your research topic.
edge_data = torch.ones([64, 64])  # initialize edge index
edge_index = coo_matrix(edge_data)
edge_index = np.vstack((edge_index.row, edge_index.col))
edge_index = torch.from_numpy(edge_index).to(torch.int64).to(device)

class GIN(torch.nn.Module):
    def __init__(self):
        super(GIN, self).__init__()

        num_features = 16  # dimension of features for each node. In our case, it's the time-steps
        dim = 32  # dimension of hidden representations
        self.dim = dim

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, 2)

    def forward(self, x, batch, edge_index=None):

        x = x.reshape([-1, 16])

        x = F.relu(self.conv1(x.float(), edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)

        x = x.view(batch, 64, self.dim)
        x = x.sum(dim=1)

        x = F.dropout(x, p=0.3, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

model = GIN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

best_acc = []
best_auc = []

# training and testing
start_time = time.perf_counter()
for epoch in range(EPOCH):
    for step, (train_x, train_y) in enumerate(train_loader):
        output = model(train_x,  BATCH_size, edge_index)
        loss = loss_func(output, train_y.long())  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    if epoch % 10 == 0:
        test_output = model(test_fea_norm1,  BATCH_size, edge_index)
        test_loss = loss_func(test_output, test_label.long())

        test_y_score = one_hot(test_label.data.cpu().numpy())  # .cup() can be removed if your device is cpu.
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
