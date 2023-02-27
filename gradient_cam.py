from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from os.path import join
import imgmodel
import os
from torchvision.models import efficientnet_v2_s
import torch
import glob
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

class ViewDataset(Dataset):
    def __init__(self, img_labels, img_dir, transform=None, target_transform=None):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.target_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                    transforms.RandomCrop((128, 128))])
        self.tensor_transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):
        ori_image = Image.open(join(self.img_dir[index])).convert("RGB")
        image = self.target_transform(ori_image)
        name = os.path.basename(os.path.abspath(os.path.join(img_dir[index], os.path.pardir)))
        label = labels_map[name]
        ori_image = self.tensor_transform(ori_image)

        return image, label, ori_image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

labels_map = {
    "kim": 0,
    "lee": 1,
    "lim": 2,
    "yeom": 3,
    "s": 4,
}
img_dir = glob.glob('./result/*/*')
e = 0

effmodel = efficientnet_v2_s(weights=None).to(device)
model = imgmodel.SkinImg(effmodel,class_num=5,dr_rate=0.2)

model_dir = glob.glob('./model256/*')

for md in model_dir:
    model.load_state_dict(torch.load(md))

# print(model.effmodel.features[5][-1])
    target_layers = [model.effmodel.features[5][-1]]
#
# for name,parameters in model.named_parameters():
#     print(name)
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=device)
    targets = None

    dataset = ViewDataset(labels_map, img_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for image, label, ori in dataloader:
        input_tensor = image
        break

#print(input_tensor.size())

    input = torch.permute(input_tensor, (0,2,3,1))
    ori = torch.permute(ori, (0,2,3,1))

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    #print(grayscale_cam.shape)
    visualization = show_cam_on_image(input.cpu().numpy()[0], grayscale_cam, use_rgb=True, image_weight = 0.5)

    plt.subplot(311)
    plt.imshow(visualization)
    plt.subplot(312)
    plt.imshow(input.cpu().numpy()[0])
    plt.subplot(313)
    plt.imshow(np.array(ori)[0])
    # plt.show()
    save_name = './gradcam_img/result_{}.png'.format(e + 20)
    plt.savefig(save_name)
    e = e + 20
