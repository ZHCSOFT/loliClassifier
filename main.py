import torch
from torch import nn, utils
from torchsummary import summary
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import PIL.Image as Image
from glob import glob
from tqdm.auto import tqdm
from torch.utils.data import random_split
from sklearn.metrics import classification_report, confusion_matrix
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, save_path='.'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_path (str): Modifies checkpoint's location
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # Saves model when validation loss decrease.
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.save_path + '/checkpoint.pt')
        torch.save(model, self.save_path + '/best_model.pkl')
        self.val_loss_min = val_loss


class ImgDataset(utils.data.Dataset):
    def __init__(self, paths, img_size, all_class, is_train):
        self.paths = paths
        self.all_class = all_class
        self.class_map = {cls: int(i) for i, cls in enumerate(all_class)}
        self.num_classes = len(self.class_map)
        self.img_size = img_size
        self.is_train = is_train

    def __getitem__(self, index):
        transform_set = [transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5)),
                         transforms.RandomRotation(90),
                         transforms.RandomHorizontalFlip(p=0.5),
                         transforms.RandomVerticalFlip(p=0.5),
                         transforms.RandomPerspective(distortion_scale=0.5, p=0.5,
                                                      fill=0),
                         # transforms.GaussianBlur(2, sigma=(0.1, 2.0)), # torch 1.7 only
                         transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5, saturation=0.5)
                         ]
        if self.is_train:
            img_transforms = transforms.Compose([transforms.Resize(size=self.img_size),
                                                 transforms.RandomOrder(transform_set)
                                                 ])
        else:
            img_transforms = transforms.Resize(size=self.img_size)

        path = self.paths[index]
        # img = cv2.imread(path)[:, :, ::-1]
        # img = cv2.resize(img, self.img_size)
        # normalize to 0~1
        # img = img / 255.
        # (H, W, C) -> (C, H, W)
        # img = np.moveaxis(img, -1, 0)
        img = Image.open(path).convert('RGB')
        img = img_transforms(img)
        img = np.asarray(img)
        img = np.moveaxis(img, -1, 0)
        # read class label
        class_pneumonia = path.split(os.sep)[-1].split('_')[-2]
        class_index = class_map[class_pneumonia]

        img_transforms = transforms.Compose([  # transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            #                      std=[0.1, 0.1, 0.1]),
                                             transforms.RandomErasing(p=0.5, scale=(0.02, 0.33),
                                                                      ratio=(0.3, 3.3),
                                                                      value=0, inplace=False),
                                             ])
        img = torch.from_numpy(img.copy()).float()
        img = img_transforms(img)
        class_index = torch.tensor(int(class_index))

        return img, class_index

    def __len__(self):
        return len(self.paths)


str_time = str(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
filePath = './' + str_time + '_loli_binary'

if not os.path.exists(filePath):
    os.makedirs(filePath)

fp = open(filePath + '/record.txt', 'a')

image_size = (224, 224)
batch_size = 48
learning_rate = 0.0001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['Explicit', 'Safe']
#  'normal': 0, 'bacteria': 1, 'virus': 2

fp.writelines('batch_size = '+str(batch_size) + ' ' + 'learning rate = ' + str(learning_rate))

class_map = {cls: i for i, cls in enumerate(classes)}
print(class_map)

dataset_file_paths = glob('D:/Dataset/loli/*.jpg')

dataset_file_paths = [x for x in dataset_file_paths if 'Questionable' not in x]

data_set = ImgDataset(dataset_file_paths, image_size, classes, True)

train_size = int(0.8 * len(data_set))
val_size = len(data_set) - train_size

train_set, val_set = random_split(data_set, [train_size, val_size])

train_generator = utils.data.DataLoader(train_set, batch_size, shuffle=True)
val_generator = utils.data.DataLoader(val_set, batch_size, shuffle=False)

print('Train:', train_size, 'val:', val_size)

plt.subplot(1, 2, 1), plt.title('origin_image')
plt.imshow(Image.open(dataset_file_paths[0]).convert('RGB'))
plt.subplot(1, 2, 2), plt.title('resized_image')
resize = transforms.Resize(image_size)
plt.imshow(resize(Image.open(dataset_file_paths[0]).convert('RGB')))
plt.show()
'''
model = models.vgg19(pretrained=True).to(device)
model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(4096, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(p=0.2),
                                 nn.Linear(4096, 3)).to(device)
'''
model = models.resnet50(pretrained=True).to(device)

fc_features = model.fc.in_features
model.fc = nn.Sequential(nn.Dropout(0.5),
                         nn.Linear(fc_features, len(classes)).to(device))

summary(model, (3, image_size[0], image_size[1]))

epochs = 300

# early stopping patience; how long to wait after last time validation loss improved.
patience = 25

loss_func = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

logs = {'train': [], 'val': []}

early_stopping = EarlyStopping(patience=patience, verbose=True, save_path=filePath)

for epoch in range(epochs):
    train_acc = 0.
    train_loss = 0.
    val_acc = 0.
    val_loss = 0

    # training
    model.train()  # set to training mode
    for i, data in enumerate(tqdm(train_generator)):
        x_train, y_train = data
        x_train, y_train = x_train.to(device), y_train.to(device)  # move to GPU
        optimizer.zero_grad()  # reset gradients
        preds = model(x_train)
        loss_value = loss_func(preds, y_train)
        _, train_pred = torch.max(preds, 1)  # get the highest probability
        # update model parameters by back propagation
        loss_value.backward()
        optimizer.step()

        train_acc += (train_pred.cpu() == y_train.cpu()).sum().item()
        train_loss += loss_value.item()

    # validation

    with torch.no_grad():
        model.eval()  # set to evaluation mode
        for i, data in enumerate(val_generator):
            x_val, y_val = data
            x_val, y_val = x_val.to(device), y_val.to(device)
            preds = model(x_val)
            loss_value = loss_func(preds, y_val)
            _, val_pred = torch.max(preds, 1)

            val_acc += (val_pred.cpu() == y_val.cpu()).sum().item()
            val_loss += loss_value.item()

    epoch_loss_train = train_loss / len(train_generator)
    epoch_loss_val = val_loss / len(val_generator)
    epoch_acc_train = train_acc / len(train_set)
    epoch_acc_val = val_acc / len(val_set)

    print(
        f'\nepoch {epoch}: train loss: {epoch_loss_train:.4f}, acc: {epoch_acc_train:.3f} | '
        f'val loss: {epoch_loss_val:.4f} val acc: {epoch_acc_val:.3f}')
    fp.writelines(
        f'\nepoch {epoch}: train loss: {epoch_loss_train:.4f}, acc: {epoch_acc_train:.3f} | '
        f'val loss: {epoch_loss_val:.4f} val acc: {epoch_acc_val:.3f}')

    logs['train'].append(epoch_loss_train)
    logs['val'].append(epoch_loss_val)

    # early_stopping needs the validation loss to check if it has decreased,
    # and if it has, it will make a checkpoint of the current model
    early_stopping(val_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

torch.save(model.state_dict(), filePath + '/model_Adam_resnet50.ckpt')

plt.plot(logs['train'])
plt.plot(logs['val'])
plt.legend(['train', 'val'])

plt.title('loss')
plt.savefig(filePath + '/curve.png')
plt.show()
fp.close()

y_pred = np.array([])
y_true = np.array([])

model.eval()

with torch.no_grad():
    for i, data in enumerate(tqdm(val_generator)):
        x_val, y_val = data
        x_val = x_val.to(device)
        pred = model(x_val)
        _, pred = torch.max(pred, 1)
        y_pred = np.append(y_pred, pred.cpu().numpy())
        y_true = np.append(y_true, y_val)

        print(confusion_matrix(y_true, y_pred))
