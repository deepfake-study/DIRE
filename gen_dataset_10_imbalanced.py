import random
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
import eagerpy as ep
from foolbox import PyTorchModel, accuracy, samples
import foolbox.attacks as fa
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.models as models
import torch

# Define the transform to crop and resize the image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the ImageNet validation dataset
dataset = ImageNet(root='/home/DATA/ITWM/lorenzp/ImageNet', split='train', transform=transform)

# Get the list of all class labels
class_labels = dataset.classes

# Select 10 random classes
random_classes = random.sample(class_labels, 10)

model = models.resnet18(pretrained=True).eval()
preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

attack = fa.LinfPGD()
epsilons = 8/255.; eps = "8255"
# epsilons = 4/255.; eps = "4255"

dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

counter = 0
for i, (image, label) in enumerate(dataloader):
    if class_labels[label] in random_classes:
        raw_advs, clipped_advs, success = attack(fmodel, image.cuda(), torch.tensor(label).cuda(), epsilons=epsilons)

        if success.cpu().item():
            clipped_advs = clipped_advs.cpu()[0]
            counter += 1
            image_pil = transforms.ToPILImage()(image.cpu()[0])
            image_pil.save(f'data/adversarial/images_{eps}/nor/{counter}.png')
            image_pil_adv = transforms.ToPILImage()(clipped_advs)
            image_pil_adv.save(f'data/adversarial/images_{eps}/att/{counter}.png')

    if counter >= 1000:
        break