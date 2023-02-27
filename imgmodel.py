import torch
from torch import nn, device
from PIL import Image
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import torchvision.transforms as transforms
from torchvision.models import efficientnet_v2_s
import os
from os.path import join
import glob
import pandas as pd
from transformers.optimization import get_cosine_schedule_with_warmup


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    transforms.RandomCrop((128, 128))])


class ImageDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform=None, target_transform=None):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):
        image = Image.open(join(self.img_dir[index])).convert("RGB")
        image = transform(image)
        name = os.path.basename(os.path.abspath(os.path.join(img_dir[index], os.path.pardir)))
        label = labels_map[name]

        return image, label


labels_map = {
    "kim": 0,
    "lee": 1,
    "lim": 2,
    "yeom": 3,
    "s": 4,
}
img_dir = glob.glob('./result/*/*')

ds = ImageDataset(labels_map, img_dir, transform)
dl = DataLoader(ds, batch_size=1, shuffle=True)

# for img, label in dl:
#     print(img)
#     print(label)
#     break

effmodel = efficientnet_v2_s(weights=None).to(device)


class SkinImg(nn.Module):
    def __init__(self,
                 effmodel,
                 class_num=5,
                 dr_rate=0.2,
                 ):
        super(SkinImg, self).__init__()
        self.effmodel = effmodel
        # 이거 모양을 확인해봐야 다음 레이어 만들 수 있음->batch,1000
        dnn = [nn.Linear(1000, 256),
               nn.ReLU(),
               nn.Linear(256, 16),
               nn.ReLU(),
               nn.Linear(16, class_num)]
        self.fc_layer = nn.Sequential(*dnn)

        self.dropout = nn.Dropout(p=dr_rate)

        self.softmax = nn.Softmax()

    def forward(self, img):
        out = self.effmodel(img)
        out = self.dropout(out)
        out = self.fc_layer(out)
        out = self.softmax(out)

        return out


# imgmodel = SkinImg()
# img = imgmodel(img)
# print(img.shape)


def calc_accuracy(X, Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy() / max_indices.size()[0]
    return train_acc



def train(class_num=5,
          dr_rate=0.2,
          batch_size=16,
          learning_rate=2e-5, #g
          num_epochs=800,
          warmup_ratio=0.05,
          save_name='result'
          ):
    # 1. 데이터로더 인스턴스 생성
    dataset = ImageDataset(labels_map, img_dir, transform)

    test_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - test_size

    testset, trainset = random_split(dataset, [test_size, train_size],
                                     generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # 2. 모델 인스턴스 생성
    imgmodel = SkinImg(effmodel,
                       dr_rate=dr_rate,
                       class_num=class_num).to(device)

    # 손실, 최적화 인스턴스 생성
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(imgmodel.parameters(), lr=learning_rate)

    total_step = len(train_dataloader) * num_epochs
    warmup_step = int(total_step * warmup_ratio)
    schedular = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_step)

    # epoch만큼 반복 학습 구성
    train_losses = {'step': [], 'losses': []}
    test_losses = {'epoch': [], 'losses': [], 'acc': []}

    for e in range(num_epochs):
        imgmodel.train()

        train_loss = 0.0
        for input, label in train_dataloader:
            optimizer.zero_grad()
            output = imgmodel(input.to(device))
            loss = loss_fn(output, label.to(device))
            loss.backward()
            optimizer.step()
            schedular.step()
            train_loss += loss.data.cpu().numpy()

        print("epoch {} train loss {}".format(e + 1, train_loss / len(train_dataloader)))

        imgmodel.eval()
        test_acc = 0
        test_loss = 0
        with torch.no_grad():
            for input, label in test_dataloader:
                out = imgmodel(input.to(device))
                test_loss += loss_fn(out, label.to(device)).data.cpu().numpy()
                test_acc += calc_accuracy(out, label.to(device))

        print("epoch {} test loss {} test acc {}".format(e + 1, test_loss / len(test_dataloader),
                                                         test_acc / len(test_dataloader)))

        mname = '{}_{}'.format(save_name, e + 1)
        mpath = './model/{}'.format(mname)
        lpath = './log/{}'.format(save_name)
        pd.DataFrame(train_losses).to_csv('{}_train.csv'.format(lpath), index=False)
        pd.DataFrame(test_losses).to_csv('{}_test.csv'.format(lpath), index=False)
        torch.save(imgmodel.state_dict(), mpath)


# train()