from torchvision import transforms


class Transforms:
    def __init__(self, cfg):
        self.cfg = cfg

    def get_transforms(self, mode='train'):
        if mode == 'train':
            return transforms.Compose([transforms.ToPILImage(),
                                       transforms.RandomResizedCrop(self.cfg.crop, scale=tuple(self.cfg.scale)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=self.cfg.mean,
                                                            std=self.cfg.std)
                                       ])
        else:
            return transforms.Compose([transforms.ToPILImage(),
                                       transforms.Resize((self.cfg.crop, self.cfg.crop)),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=self.cfg.mean,
                                                            std=self.cfg.std)
                                       ])

    def get_inverse_transform(self):
        return transforms.Normalize(mean=[-m/s for m, s in zip(self.cfg.mean, self.cfg.std)],
                                    std=[1/s for s in self.cfg.std])
