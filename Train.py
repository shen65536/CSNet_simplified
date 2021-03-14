import os
import time
import torch
import argparse
import PIL.Image
import torchvision
import numpy as np
import torch.nn as nn
import scipy.io as scio
import torch.optim as optimize
import torch.utils.data as data


from CSNet import CSNet


parser = argparse.ArgumentParser(description="Demo of CSNet.")
parser.add_argument("--epochs", default=100, type=int, metavar="NUM")
parser.add_argument("--batch_size", default=20, type=int, metavar="SIZE")
parser.add_argument("--block_size", default=32, type=int, metavar="SIZE")
parser.add_argument("--test_path", default="./BSD100", metavar="PATH")
parser.add_argument("--train_path", default="./images/train", metavar="PATH")
parser.add_argument("--save_file", default="./CSNet.pth", metavar="FILE")
opt = parser.parse_args()


def get_files(path):
    fs = []
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            _, extension = os.path.splitext(name)
            if extension == ".jpg":
                fs.append(os.path.join(root, name))
    return fs


def crop(x, BlockSize=32):
    h = x.size(1)
    w = x.size(2)
    nc = x.size(0)

    ind1 = range(0, h, BlockSize)
    ind2 = range(0, w, BlockSize)
    y = torch.zeros(len(ind1) * len(ind2), nc, BlockSize, BlockSize)

    count = 0
    for i in ind1:
        for j in ind2:
            temp = x[:, i:i + BlockSize, j:j+BlockSize]
            y[count, :, :, :, ] = temp
            count = count + 1
    return y


class DataSet(data.Dataset):
    def __init__(self, root):
        super(DataSet, self).__init__()
        self.fs = get_files(root)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.Grayscale(num_output_channels=1)
        ])

    def __len__(self):
        return len(self.fs)

    def __getitem__(self, index):
        path = self.fs[index]
        image = PIL.Image.open(path)
        image = self.transforms(image)
        items = crop(image)

        return items


def loader():
    train_dataset = DataSet(opt.train_path)
    train_data_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    return train_data_loader


def train(net):
    if not os.path.isfile(opt.save_file):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)

        train_set = loader()

        criterion = nn.MSELoss()
        optimizer = optimize.Adam(net.parameters())
        scheduler = optimize.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

        print("=> Train start.")
        time_start = time.time()

        for epoch in range(opt.epochs):
            for i, input in enumerate(train_set):
                for item in range(input.size(1)):
                    data = input[:, item, :, :, :]
                    net.train()
                    optimizer.zero_grad()

                    output = net(data)

                    loss1 = criterion(output, data)
                    loss1.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    use_time = time.time() - time_start
                    print("=> epoch: {}, batch: {:.0f}, loss1: {:.4f}, lr: {}, time: {:.3f}"
                          .format(epoch + 1, i + 1, loss1.item(), optimizer.param_groups[0]["lr"], use_time))
            scheduler.step()
        print("=> Train end.")
        torch.save(net.state_dict(), opt.save_file)


def val(net):
    if os.path.isfile(opt.save_file):
        if not os.path.exists("./images"):
            os.makedirs("./images")

        net.load_state_dict(torch.load(opt.save_file, map_location='cpu'))
        tensor2image = torchvision.transforms.ToPILImage()

        for i in range(100):
            name = (opt.test_path + "/({}).mat".format(i + 1))
            mat = scio.loadmat(name)
            mat = mat['temp3']
            tensor = torch.from_numpy(np.array(mat))

            output = torch.zeros(tensor.size())
            h, w = tensor.size()
            num_h = h / opt.block_size
            num_w = w / opt.block_size
            for idx_h in range(int(num_h)):
                for idx_w in range(int(num_w)):

                    h1 = idx_h * opt.block_size
                    h2 = h1 + opt.block_size
                    w1 = idx_w * opt.block_size
                    w2 = w1 + opt.block_size

                    input = tensor[h1:h2, w1:w2].unsqueeze(0).unsqueeze(0)
                    tmp = net(input)
                    output[h1:h2, w1:w2] = tmp
            image = tensor2image(output)
            image.save("./res/{}.jpg".format(i + 1))
            print("=> image {} done!".format(i))


if __name__ == "__main__":
    cs_net = CSNet()

    train(cs_net)
    val(cs_net)
