from torchvision.io import read_image
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.functional import conv2d
import matplotlib.pyplot as plt


LABEL = "NE RABOTAET"


def forward(model, x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)

    return x


def main():
    path = 'MyDinosaurAlphabet.jpeg'

    dinosaurs = read_image(path).float() / 255
    dinosaurs_shape = dinosaurs.size()

    image = plt.imread(path)
    plt.imsave('temp_' + path, image)

    plt.imshow(dinosaurs.permute(1, 2, 0))
    plt.show()

    dinosaur = dinosaurs[:, 470:745, 265:425]

    plt.imshow(dinosaur.permute(1, 2, 0))
    plt.show()

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    small_cube = forward(model, dinosaur[None, :, :, :])

    for label in ['Top 1', 'Top 2', 'Top 3']:
        dinosaurs = read_image('temp_' + path).float() / 255
        big_cube = forward(model, dinosaurs[None, :, :, :])
        heatmap = conv2d(big_cube, small_cube, padding='same').squeeze().detach()
        top = heatmap.argmax()
        heatmap_shape = heatmap.size()
        k_x = dinosaurs_shape[2] / heatmap_shape[1]
        k_y = dinosaurs_shape[1] / heatmap_shape[0]
        x = int((top % dinosaurs_shape[2]) * k_x)
        y = int(top / dinosaurs_shape[2] * k_y)
        circle = plt.Circle((x, y), 3, color='b')
        plt.imshow(dinosaurs.permute(1, 2, 0))
        plt.gca().add_patch(circle)
        plt.text(x - 2, y + 1, label)
        plt.imsave('temp_' + path, heatmap)

    plt.imshow(heatmap, cmap='autumn')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
