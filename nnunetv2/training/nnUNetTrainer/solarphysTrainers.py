import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

class nnUNetTrainer_50epochs_tempval(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 50
        self.temporal_cross_val = True


class nnUNetTrainer_75epochs_tempval(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 75
        self.temporal_cross_val = True


class nnUNetTrainer_100epochs_tempval(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True


class nnUNetTrainer_150epochs_tempval(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True


class nnUNetTrainer_200epochs_tempval(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
        self.temporal_cross_val = True


class nnUNetTrainer_250epochs_tempval(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.temporal_cross_val = True


class nnUNetTrainer_1000epochs_tempval(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.temporal_cross_val = True


class nnUNetTrainer_1000epochs_crossval(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.temporal_cross_val = False


class Trainer_100epochs_1_2expo(nnUNetTrainer):

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.2)
        return optimizer, lr_scheduler
    

class Trainer_100epochs_1_4expo(nnUNetTrainer):

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.4)
        return optimizer, lr_scheduler


class Trainer_100epochs_1_6expo(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.6)
        return optimizer, lr_scheduler
    

class Trainer_100epochs_2_0expo(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.0)
        return optimizer, lr_scheduler
    

class Trainer_100epochs_2_4expo(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.4)
        return optimizer, lr_scheduler
    

class Trainer_100epochs_0_7expo(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.7)
        return optimizer, lr_scheduler


class Trainer_100epochs_2_0expo_2_0ilr_(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True
        self.initial_lr = 0.02

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.0)
        return optimizer, lr_scheduler


class Trainer_100epochs_2_4expo_2_0ilr_(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True
        self.initial_lr = 0.02  

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.4)
        return optimizer, lr_scheduler
    

class Trainer_100epochs_2_8expo_2_0ilr_(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True
        self.initial_lr = 0.02  

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.8)
        return optimizer, lr_scheduler


class Trainer_150epochs_1_2expo(nnUNetTrainer):

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.2)
        return optimizer, lr_scheduler
    

class Trainer_150epochs_1_4expo(nnUNetTrainer):

    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.4)
        return optimizer, lr_scheduler


class Trainer_150epochs_1_6expo(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.6)
        return optimizer, lr_scheduler
    

class Trainer_150epochs_2_0expo(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.0)
        return optimizer, lr_scheduler


class Trainer_150epochs_2_4expo(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.4)
        return optimizer, lr_scheduler
    

class Trainer_150epochs_2_0expo_1_5ilr_(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True
        self.initial_lr = 0.015  

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.0)
        return optimizer, lr_scheduler


class Trainer_150epochs_2_4expo_1_5ilr_(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True
        self.initial_lr = 0.015  

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.4)
        return optimizer, lr_scheduler
    

class Trainer_150epochs_2_8expo_1_5ilr_(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True
        self.initial_lr = 0.015  

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.8)
        return optimizer, lr_scheduler
    

class Trainer_150epochs_3_5expo_1_5ilr_(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True
        self.initial_lr = 0.015  

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=3.5)
        return optimizer, lr_scheduler


class Trainer_150epochs_4_0expo_1_5ilr_(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True
        self.initial_lr = 0.015  

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=4.0)
        return optimizer, lr_scheduler


class Trainer_150epochs_4_5expo_1_5ilr_(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True
        self.initial_lr = 0.015  

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=4.5)
        return optimizer, lr_scheduler






# Trainers with a dice and CE weight difference:
class Trainer_150epochs_0_8expo_1_5ilr_2_1weights(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True
        self.initial_lr = 0.015
        self.weight_ce = 1
        self.weight_dice = 2  
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=0.8)
        return optimizer, lr_scheduler
    


class Trainer_150epochs_2_0expo_1_5ilr_2_1weights(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True
        self.initial_lr = 0.015
        self.weight_ce = 1
        self.weight_dice = 2  
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.0)
        return optimizer, lr_scheduler
    


class Trainer_150epochs_4_0expo_1_5ilr_2_1weights(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True
        self.initial_lr = 0.015
        self.weight_ce = 1
        self.weight_dice = 2  
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=4.0)
        return optimizer, lr_scheduler

    


class Trainer_100epochs_2_0expo_2_0ilr_2_1weights(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True
        self.initial_lr = 0.02
        self.weight_ce = 1
        self.weight_dice = 2
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.0)
        return optimizer, lr_scheduler




class Trainer_150epochs_4_0expo_1_5ilr_3_1weights(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True
        self.initial_lr = 0.015
        self.weight_ce = 1
        self.weight_dice = 3  
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=4.0)
        return optimizer, lr_scheduler

    


class Trainer_100epochs_2_0expo_2_0ilr_3_1weights(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True
        self.initial_lr = 0.02
        self.weight_ce = 1
        self.weight_dice = 3
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.0)
        return optimizer, lr_scheduler





class Trainer_150epochs_4_0expo_1_5ilr_4_1weights(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = True
        self.initial_lr = 0.015
        self.weight_ce = 1
        self.weight_dice = 4
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=4.0)
        return optimizer, lr_scheduler

    


class Trainer_100epochs_2_0expo_2_0ilr_4_1weights(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True
        self.initial_lr = 0.02
        self.weight_ce = 1
        self.weight_dice = 4
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.0)
        return optimizer, lr_scheduler

    


class Trainer_100epochs_1_4expo_1_0ilr_2_1weights(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True
        self.initial_lr = 0.01
        self.weight_ce = 1
        self.weight_dice = 2
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.4)
        return optimizer, lr_scheduler
    
    


class Trainer_100epochs_1_4expo_1_0ilr_1_2weights(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = True
        self.initial_lr = 0.01
        self.weight_ce = 2
        self.weight_dice = 1
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=1.4)
        return optimizer, lr_scheduler



class Trainer_100epochs_2_0expo_2_0ilr_2_1weights_crossval(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.temporal_cross_val = False
        self.initial_lr = 0.02
        self.weight_ce = 1
        self.weight_dice = 2
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.0)
        return optimizer, lr_scheduler
    


class Trainer_150epochs_2_0expo_1_5ilr_2_1weights_crossval(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json,
                 device=torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 150
        self.temporal_cross_val = False
        self.initial_lr = 0.015
        self.weight_ce = 1
        self.weight_dice = 2  
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay, momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs, exponent=2.0)
        return optimizer, lr_scheduler
    

### Implementation of the instance based F(beta=075) loss component

class nnUNetTrainer_1000epochs_crossval_with_ifb_loss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.temporal_cross_val = False
        self.is_custom_loss_function = True
        #self.weight_ifb = 1
        #self.beta_ifb=0.75


class nnUNetTrainer_1000epochs_tempval_with_ifb_loss(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 1000
        self.temporal_cross_val = True
        self.is_custom_loss_function = True
        #self.weight_ifb = 1
        #self.beta_ifb=0.75