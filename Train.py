import os
import time
import torch
import argparse
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
parser.add_argument("--test_path", default="../BSD100", metavar="PATH")
parser.add_argument("--train_path", default="../images/train", metavar="PATH")
opt = parser.parse_args()


def loader():
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomCrop(opt.block_size),
        torchvision.transforms.Grayscale(num_output_channels=1)
    ])

    train_dataset = torchvision.datasets.ImageFolder(opt.train_path, transform=transforms)
    train_data_loader = data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    return train_data_loader


def train(net):
    if not os.path.isfile("./CSNet.pth"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net.to(device)

        train_set = loader()

        criterion = nn.MSELoss()
        optimizer = optimize.Adam(net.parameters())
        scheduler = optimize.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.1)

        print("=> Train start.")
        time_start = time.time()

        for epoch in range(opt.epochs):
            for i, (input, name) in enumerate(train_set):
                optimizer.zero_grad()
                output = net(input)
                loss = criterion(output, input)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                use_time = time.time() - time_start

                print("=> epoch: {}, batch: {:.0f}, loss: {:.4f}, lr: {}, time: {:.3f}"
                      .format(epoch + 1, i + 1, loss.item(), optimizer.param_groups[0]["lr"], use_time))
            scheduler.step()
        print("=> Train end.")
        torch.save(net.state_dict(), net.save_path)


def val(net):
    if os.path.isfile("./CSNet.pth"):
        if not os.path.exists("./images"):
            os.makedirs("./images")

        net.load_state_dict(torch.load('./CSNet.pth', map_location='cpu'))
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

                    input = tensor[h1:h2, w1:w2]
                    input = input.unsqueeze(0).unsqueeze(0)
                    tmp = net(input)
                    output[h1:h2, w1:w2] = tmp
            image = tensor2image(output)
            image.save("./images/{}.jpg".format(i + 1))


if __name__ == "__main__":
    cs_net = CSNet()
    train(cs_net)
    val(cs_net)
