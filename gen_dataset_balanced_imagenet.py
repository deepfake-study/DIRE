import os
from datetime import datetime

import argparse
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
import time
import timm

import cfg

from misc import (
    args_handling,
    print_args,
    save_to_pt,
    convert_to_float,
    create_dir,
    str2bool,
    create_log_file,
    save_log
)

DEEPFOOL = ['fgsm', 'bim', 'pgd', 'df', 'cw']
AUTOATTACK = ['aa', 'apgd-ce', 'square']

def main() -> None:
    parser = argparse.ArgumentParser("gen")
    parser.add_argument("--run_nr",  default="run_1", help="")
    parser.add_argument("--att",  default="pgd", choices=[None, 'fgsm', 'bim', 'pgd', 'df', 'cw'], help="")
    parser.add_argument("--dataset",  default="imagenet", choices=['imagenet', 'cifar10', 'cifar100'], help="")
    parser.add_argument("--model",  default="resnet18", choices=['wrn50-2', 'wrn28-10', 'vgg16'], help="")
    # parser.add_argument("--save_nor",  default="normalos.pt", help="")
    # parser.add_argument("--save_adv",  default="adverlos.pt", help="")
    parser.add_argument("--eps",  default="8/255", help="")
    parser.add_argument("--norm",  default="Linf", choices=['Linf', 'L2', 'L1'], help="")
    parser.add_argument("--version",  default="standard", help="")
    parser.add_argument("--bs",   default=1, help="")
    parser.add_argument("--classes",   default=100, help="")
    parser.add_argument("--max_samples_per_class",   default=100, help="")
    # parser.add_argument("--max_counter",  default=2000, help="")
    # parser.add_argument("--debug",  default=True, type=str2bool, help="")
    # parser.add_argument("--shuffle",  default=False, type=str2bool, help="")
    parser.add_argument("--benign",  default=True, type=str2bool, help="")
    parser.add_argument("--attack",  default=True, type=str2bool, help="")

    parser.add_argument('--save_json', default="", help='Save settings to file in json format. Ignored in json file')
    parser.add_argument('--load_json', default="", help='Load settings from file in json format. Command line options override values in file.')

    args = parser.parse_args()
    args = args_handling(args, parser, "./configs/gen")
    eps = args.eps
    args.eps = convert_to_float(args.eps)
    print_args(args)

    print("Create paths!")
    base_pth = os.path.join('./data/gen', args.run_nr, args.dataset, args.model, args.att)
    create_dir(base_pth)
    log_pth = os.path.join(base_pth, 'logs')
    log = create_log_file(args, log_pth)
    log['timestamp_start'] =  datetime.now().strftime("%Y-%m-%d-%H-%M")

    # Define the transform to crop and resize the image
    transform = transforms.Compose([
        transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ])

    # Load the ImageNet train dataset
    dataset = ImageNet(root=os.path.join(cfg.DATASET_BASE_PTH, 'ImageNet'), split='train', transform=transform)

    # Get the list of all class labels
    class_labels = dataset.classes

    # Select 10 random classes
    random_classes = random.sample(class_labels, args.classes)
    max_samples_per_class = args.max_samples_per_class

    model = timm.create_model(args.model, pretrained=True).eval()

    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    if args.att == 'pgd':
        attack = fa.LinfPGD()
    elif args.att == 'df':
        attack = fa.L2DeepFoolAttack()
        args.eps = None
    elif args.att in AUTOATTACK:
        breakpoint()
        from helper_gen.attacks.sub_autoattack.auto_attack import AutoAttack as AutoAttack_mod
        # https://colab.research.google.com/drive/1uZrW3Sg-t5k6QVEwXDdjTSxWpiwPGPm2?usp=sharing#scrollTo=jYnKIzXAgV4W
        adversary = AutoAttack_mod(fmodel, norm=args.norm.capitalize(), eps=args.eps, 
                                    log_path=os.path.join(log_pth, args.load_json.split('/')[-1]).replace("json", "log"),  verbose=args.debug, version=args.version)
        if args.version == 'individual':
            adversary.attacks_to_run = [ args.att ]
        adversary.seed = 0 # every attack is seeded by 0. Otherwise, it would be randomly seeded for each attack.


    start_time = time.time()

    # Create a generator function for loading images
    seen_indices = set()

    def image_loader():
        # dataset = ImageNet(root='/home/DATA/ITWM/lorenzp/ImageNet', split='val', transform=transform)
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        for index in indices:
            if index not in seen_indices:
                seen_indices.add(index)
                image, label = dataset[index]
                yield image, label

    if args.benign:
        folder_path = f'data/gen/{args.run_nr}/{args.dataset}/{args.model}/{args.att}/images_{eps}/ben_{args.classes}/'
        print("benign", folder_path)
        os.makedirs(folder_path, exist_ok=True)
        class_counts = {label: 0 for label in random_classes}
        # Create a set to keep track of seen indices
        counter = 0
        for i, (image, label) in enumerate(image_loader(), 1):
            if class_labels[label] in random_classes:
                class_counts[class_labels[label]] += 1
                if class_counts[class_labels[label]] <= max_samples_per_class:
                    img = image.cuda().unsqueeze(dim=0)
                    lab = torch.tensor(label).cuda().unsqueeze(dim=0)
                    y_pred = torch.max(fmodel(img), dim=1)[1].item()
                    counter += 1
                    if counter % 100 ==0:
                        print("counter: ", counter)
                        # breakpoint()
                    image_pil = transforms.ToPILImage()(img.cpu()[0])
                    image_pil.save(os.path.join(folder_path, f'{counter}.png'))
            
            if all(count >= max_samples_per_class for count in class_counts.values()):
                print("Breakpoint: num max classes filled ")
                break

    if args.attack:
        folder_path = f'data/gen/{args.run_nr}/{args.dataset}/{args.model}/{args.att}/images_{eps}/att_{args.classes}/'
        print("attack", folder_path)
        os.makedirs(folder_path, exist_ok=True)
        class_counts = {label: 0 for label in random_classes}

        counter = 0
        all_counter = 0
        for i, (image, label) in enumerate(image_loader(), 1):
            if class_labels[label] in random_classes:
                if class_counts[class_labels[label]] < max_samples_per_class:
                    img = image.cuda().unsqueeze(dim=0)
                    lab = torch.tensor(label).cuda().unsqueeze(dim=0)
                    y_pred = torch.max(fmodel(img), dim=1)[1].item()
                    # print(y_pred, label, y_pred == label)
                    if y_pred == label:
                        all_counter += 1
                        if args.att in DEEPFOOL:
                            raw_advs, clipped_advs, success = attack(fmodel, img, lab, epsilons=args.eps)

                        if args.att in AUTOATTACK:
                            if args.version == 'standard':
                                x_, y_, max_nr, success = adversary.run_standard_evaluation(img, lab, bs=args.bs, return_labels=True)
                            else: 
                                adv_complete = adversary.run_standard_evaluation_individual(img, lab, bs=args.bs, return_labels=True)
                                x_, y_, max_nr, success = adv_complete[ args.att ]  

                        if success.cpu().item():
                            class_counts[class_labels[label]] += 1
                            clipped_advs = clipped_advs.cpu()[0]
                            counter += 1
                            if counter % 100 == 0:
                                print("counter: ", counter)
                            image_pil_adv = transforms.ToPILImage()(clipped_advs)
                            image_pil_adv.save(os.path.join(folder_path, f'{counter}.png'))
            
            if all(count >= max_samples_per_class for count in class_counts.values()):
                print("Breakpoint: num max classes filled ")
                break
            
        asr = all_counter / counter

    print("counter", counter)
    count = sum(1 for c in class_counts.values() if c >= max_samples_per_class)
    print(count)

    # Calculate the elapsed time
    elapsed_time = time.time() - start_time
    # Convert elapsed time to hours and minutes
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)

    log['elapsed_time'] = str(hours) + "h:" + str(minutes) + "m"

    log['final_nr_samples'] = counter
    log['asr'] = round(asr,4)
    # log['clean_acc'] = 0 if len(clean_acc_list) == None else round(np.mean(clean_acc_list), 4)

    save_log(args, log, log_pth)

if __name__ == "__main__":
    main()