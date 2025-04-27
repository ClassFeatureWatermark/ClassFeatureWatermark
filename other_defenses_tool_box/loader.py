import torch
from torchvision import transforms
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

class Box():
    def __init__(self, opt, root_path) -> None:
        self.opt = opt
        self.dataset = opt.dataset
        self.tlabel = opt.tlabel
        self.attack = opt.poison_type
        self.normalizer = self.get_normalizer()
        self.denormalizer = self.get_denormalizer()
        self.size = opt.size
        self.attack_type = opt.attack_type
        self.root = root_path

        if self.attack_type == "all2all":
            self.res_path = self.attack + "-" + "-targetall"
        elif self.attack_type == "all2one":
            self.res_path = self.attack + "-"  + "-target" + str(self.tlabel)

    def get_save_path(self):
        save_path = os.path.join(self.root, "results/"+self.res_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        return save_path


    def get_normalizer(self):
        if 'cifar' in self.dataset:
            return transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        elif self.dataset == "gtsrb":
            return transforms.Normalize([0, 0, 0], [1, 1, 1])
        elif self.dataset == "Imagenette":
            return transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            raise Exception("Invalid dataset")
        
    def get_denormalizer(self):
        if 'cifar' in self.dataset:
            return transforms.Normalize([-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010], [1/0.2023, 1/0.1994, 1/0.2010])
        elif self.dataset == "gtsrb":
            return transforms.Normalize([0, 0, 0], [1, 1, 1])
        elif self.dataset == "Imagenette":
            return transforms.Normalize([-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
        else:
            raise Exception("Invalid dataset")
        
    def get_transform(self, train):
        if train == "clean" or train == "poison":
            if self.dataset == "cifar":
                return transforms.Compose([transforms.RandomCrop(size=32, padding=4),
                                           transforms.RandomHorizontalFlip(0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
            
            elif self.dataset == "imagenet":
                return transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.RandomCrop(size=224, padding=4),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            
            elif self.dataset == "gtsrb":
                return transforms.Compose([transforms.Resize((40, 40)),
                                           transforms.RandomCrop(size=32, padding=4),
                                           transforms.RandomHorizontalFlip(0.5),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0, 0, 0], [1, 1, 1])])
            else:
                raise Exception("Invalid dataset")
        
        elif train == "test":
            if self.dataset == "cifar":
                return transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
            
            elif self.dataset == "imagenet":
                return transforms.Compose([transforms.Resize((256, 256)),
                                           transforms.CenterCrop(size=224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            elif self.dataset == "gtsrb":
                return transforms.Compose([transforms.Resize((40, 40)),
                                           transforms.CenterCrop(size=32),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0, 0, 0], [1, 1, 1])])
            else:
                raise Exception("Invalid dataset")
        
        else:
            raise Exception("Invalid train")

    # def poisoned(self, img_tensor, param1=None, param2=None):
    #     if self.attack == "badnets":
    #         mask = param1
    #         ptn = param2
    #         img_tensor = self.denormalizer(img_tensor)
    #         bd_inputs = (1-mask) * img_tensor + mask*ptn
    #         return self.normalizer(bd_inputs)
    #     elif self.attack == "blend":
    #         alpha = param1
    #         trigger = param2
    #         bd_inputs = (1-alpha) * img_tensor + alpha*self.normalizer(trigger)
    #         return bd_inputs
    #     elif self.attack == "wanet":
    #         noise_grid = param1
    #         identity_grid = param2
    #         grid_temps = (identity_grid + 0.5 * noise_grid / self.size) * 1
    #         grid_temps = torch.clamp(grid_temps, -1, 1)
    #         num_bd = img_tensor.shape[0]
    #         bd_inputs = F.grid_sample(img_tensor[:num_bd], grid_temps.repeat(num_bd, 1, 1, 1), align_corners=True)
    #         return bd_inputs
    #     elif self.attack == "ia":
    #         netG = param1
    #         netM = param2
    #         patterns = netG(img_tensor)
    #         patterns = netG.normalize_pattern(patterns)
    #         masks_output = netM.threshold(netM(img_tensor))
    #         bd_inputs = img_tensor + (patterns - img_tensor) * masks_output
    #         return bd_inputs
    #     elif self.attack == "lc":
    #         mask = param1
    #         ptn = param2
    #         img_tensor = self.denormalizer(img_tensor)
    #         bd_inputs = (1-mask) * img_tensor + mask*ptn
    #         return self.normalizer(bd_inputs)
    #     elif self.attack == "bppattack":
    #         inputs_bd = self.back_to_np_4d(img_tensor, self.opt)
    #         squeeze_num = 8
    #         inputs_bd = torch.round(inputs_bd/255.0*(squeeze_num-1))/(squeeze_num-1)*255
    #         inputs_bd = self.np_4d_to_tensor(inputs_bd,self.opt)
    #         return inputs_bd
    #
    #     else:
    #         raise Exception("Invalid attack")

    # BppAttak tools
    def back_to_np_4d(self, inputs, opt):
        if opt.dataset == "cifar":
            expected_values = [0.4914, 0.4822, 0.4465]
            variance = [0.2023, 0.1994, 0.2010]
        elif opt.dataset == "mnist":
            expected_values = [0.5]
            variance = [0.5]
        elif opt.dataset == "imagenet":
            expected_values = [0.485, 0.456, 0.406]
            variance = [0.229, 0.224, 0.225]
        elif opt.dataset in ["gtsrb","celeba"]:
            expected_values = [0,0,0]
            variance = [1,1,1]
        inputs_clone = inputs.clone()
        if opt.dataset == "mnist":
            inputs_clone[:,:,:,:] = inputs_clone[:,:,:,:] * variance[0] + expected_values[0]

        else:
            for channel in range(3):
                inputs_clone[:,channel,:,:] = inputs_clone[:,channel,:,:] * variance[channel] + expected_values[channel]

        return inputs_clone*255
    
    def np_4d_to_tensor(self, inputs, opt):
        if opt.dataset == "cifar":
            expected_values = [0.4914, 0.4822, 0.4465]
            variance = [0.2023, 0.1994, 0.2010]
        elif opt.dataset == "mnist":
            expected_values = [0.5]
            variance = [0.5]
        elif opt.dataset == "imagenet":
            expected_values = [0.485, 0.456, 0.406]
            variance = [0.229, 0.224, 0.225]
        elif opt.dataset in ["gtsrb","celeba"]:
            expected_values = [0,0,0]
            variance = [1,1,1]
        inputs_clone = inputs.clone().div(255.0)

        if opt.dataset == "mnist":
            inputs_clone[:,:,:,:] = (inputs_clone[:,:,:,:] - expected_values[0]).div(variance[0])
        else:
            for channel in range(3):
                inputs_clone[:,channel,:,:] = (inputs_clone[:,channel,:,:] - expected_values[channel]).div(variance[channel])

        return inputs_clone
    
