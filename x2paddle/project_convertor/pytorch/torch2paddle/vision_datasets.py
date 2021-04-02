import paddle

class ImageFolder(paddle.vision.datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, loader=None, is_valid_file=None):
        assert target_transform is None, "The target_transform must be None in ImageFolder."
        super().__init__(root, loader=loader, transform=transform, is_valid_file=is_valid_file)