import glob
import logging
import os
import random
import codecs
import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import webdataset
from omegaconf import open_dict, OmegaConf
from skimage.feature import canny
from skimage.transform import rescale, resize
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler, ConcatDataset
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import copy
from saicinpainting.evaluation.data import InpaintingDataset as InpaintingEvaluationDataset, \
     EvalDataset as InferEvaluationDataset, \
    OurInpaintingDataset as OurInpaintingEvaluationDataset, ceil_modulo, InpaintingEvalOnlineDataset
from saicinpainting.training.data.aug import IAAAffine2, IAAPerspective2
from saicinpainting.training.data.masks import get_mask_generator

LOGGER = logging.getLogger(__name__)


class InpaintingTrainDataset(Dataset):

    def __init__(self, indir, mask_generator, transform):
        self.in_files = list(
            glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))
        self.mask_generator = mask_generator
        self.transform = transform
        self.iter_i = 0

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        mask = self.mask_generator(img, iter_i=self.iter_i)
        self.iter_i += 1
        return dict(image=img, mask=mask)


class DecaptioningTrainDataset(Dataset):

    def __init__(self, file_list, mask_generator):
        self.ttfs = []
        for ttf in os.listdir('/ssd3/lixin41/lama-main/chinese_fonts/'):
            self.ttfs.append('/ssd3/lixin41/lama-main/chinese_fonts/' + ttf)
        self.subtitles = []
        srts = open('/ssd3/lixin41/lama-main/LaMa_test_images/srt.txt',
                    'r',
                    encoding='utf-8')
        for line in srts:
            line = line.strip()

            subtitles = []
            if 'lixin41' in line:
                conts = line.split('lixin41')
                for subtitle in conts:
                    subtitles.append(subtitle)
            else:
                subtitles.append(line)
            self.subtitles.append(subtitles)
        self.mask_generator = mask_generator
        self.subtitles_len = len(self.subtitles)
        self.iter_i = 0
        print(self.subtitles_len)
        # NOTE: Please ensure file list was save in UTF-8 coding format
        self.shuffle_seed = 0
        with codecs.open(file_list, 'r', 'utf-8') as flist:
            self.lines = [line.strip() for line in flist]
            np.random.shuffle(self.lines)

    def add_subtitle(self, img):
        src = np.array(copy.deepcopy(img))

        draw = ImageDraw.Draw(img)
        # font = ImageFont.truetype(<font-file>, <font-size>)
        ttf = self.ttfs[random.randint(0, len(self.ttfs) - 1)]
        #font = ImageFont.truetype(ttf, 80) #, encoding="lat1")
        font = ImageFont.truetype(ttf, random.randint(25,
                                                      80))  #, encoding="lat1")
        text = self.subtitles[random.randint(0, self.subtitles_len - 1)]
        #text = ['锛堝叓灏忔檪鍓嶏級']
        w, h = img.size

        color = (random.randint(50, 255), random.randint(50, 255),
                 random.randint(50, 255))
        if random.randint(0, 1) == 1:
            color = (255, 255, 255)
        text_width = font.getsize(text[0])
        text_coordinate = [
            int((w - text_width[0]) / 2),
            random.randint(0, h - text_width[1])
        ]
        draw.text(text_coordinate, text[0], color, font=font)
        if len(text) > 1:
            text_coordinate[1] = text_coordinate[1] + int(
                random.uniform(1, 2) * text_width[1])
            draw.text(text_coordinate, text[1], color, font=font)
        img = np.array(img)
        grt = cv2.absdiff(src, img)
        grt = cv2.cvtColor(grt, cv2.COLOR_BGR2GRAY)
        _, grt = cv2.threshold(grt, 10, 1.0, cv2.THRESH_BINARY)
        mask = grt.copy()
        if random.randint(0, 2) == 0:
            kernel_size = 2 * random.randint(0, 1) + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            grt = cv2.erode(grt, kernel)
        else:
            kernel_size = 2 * random.randint(0, 4) + 1
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            grt = cv2.dilate(grt, kernel)
        #cv2.imwrite('img.jpg', img)
        #cv2.imwrite('src.jpg', src)
        return src, img, grt, mask

    def load_subtitle_img(self, line):
        img_path = line.strip()
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path)
        if img is None:
            raise Exception(
                "Empty image, source image path: {}".format(img_path))
        return img, img_name

    def random_crop(self, img, size=(512, 256)):
        width, height = size
        W, H = img.size[0], img.size[1]
        if H < height or W < width:
            raise Exception("Img height or width is not enough to crop")
        crop_y = random.randint(0, H - height)
        crop_x = random.randint(0, W - width)
        return img.crop((crop_x, crop_y, crop_x + width, crop_y + height))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, item):
        img, img_name = self.load_subtitle_img(self.lines[item])
        img = self.random_crop(img)
        src, img, mask, mask2 = self.add_subtitle(img)
        img = np.transpose(img, (2, 0, 1)).astype('float32') / 255.0
        src = np.transpose(src, (2, 0, 1)).astype('float32') / 255.0
        mask = mask[np.newaxis, :, :]
        mask2 = mask2[np.newaxis, :, :]
        # TODO: maybe generate mask before augmentations? slower, but better for segmentation-based masks
        #mask = self.mask_generator(img, iter_i=self.iter_i)
        noise1 = self.mask_generator(img, iter_i=self.iter_i)
        noise2 = np.random.rand(1, 256, 512)
        noise2 = 1.0 * (noise2 > np.random.uniform(0.85, 1.0))
        mask = mask * (1.0 - noise1) * (1.0 - noise2)

        #cv2.imwrite('mask.jpg', 255*mask[0,:,:])
        #cv2.imwrite('noise1.jpg', 255*noise1[0,:,:])
        #cv2.imwrite('noise2.jpg', 255*noise2[0,:,:])
        self.iter_i += 1
        return dict(image=src,
                    mask=mask2.astype('float32'),
                    d_image=img,
                    d_mask=mask.astype('float32'))


class InpaintingTrainWebDataset(IterableDataset):

    def __init__(self, indir, mask_generator, transform, shuffle_buffer=200):
        self.impl = webdataset.Dataset(indir).shuffle(shuffle_buffer).decode(
            'rgb').to_tuple('jpg')
        self.mask_generator = mask_generator
        self.transform = transform

    def __iter__(self):
        for iter_i, (img, ) in enumerate(self.impl):
            img = np.clip(img * 255, 0, 255).astype('uint8')
            img = self.transform(image=img)['image']
            img = np.transpose(img, (2, 0, 1))
            mask = self.mask_generator(img, iter_i=iter_i)
            yield dict(image=img, mask=mask)


class ImgSegmentationDataset(Dataset):

    def __init__(self, indir, mask_generator, transform, out_size, segm_indir,
                 semantic_seg_n_classes):
        self.indir = indir
        self.segm_indir = segm_indir
        self.mask_generator = mask_generator
        self.transform = transform
        self.out_size = out_size
        self.semantic_seg_n_classes = semantic_seg_n_classes
        self.in_files = list(
            glob.glob(os.path.join(indir, '**', '*.jpg'), recursive=True))

    def __len__(self):
        return len(self.in_files)

    def __getitem__(self, item):
        path = self.in_files[item]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.out_size, self.out_size))
        img = self.transform(image=img)['image']
        img = np.transpose(img, (2, 0, 1))
        mask = self.mask_generator(img)
        segm, segm_classes = self.load_semantic_segm(path)
        result = dict(image=img,
                      mask=mask,
                      segm=segm,
                      segm_classes=segm_classes)
        return result

    def load_semantic_segm(self, img_path):
        segm_path = img_path.replace(self.indir,
                                     self.segm_indir).replace(".jpg", ".png")
        mask = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.out_size, self.out_size))
        tensor = torch.from_numpy(np.clip(mask.astype(int) - 1, 0, None))
        ohe = F.one_hot(
            tensor.long(),
            num_classes=self.semantic_seg_n_classes)  # w x h x n_classes
        return ohe.permute(2, 0, 1).float(), tensor.unsqueeze(0)


def get_transforms(transform_variant, out_size):
    if transform_variant == 'default':
        transform = A.Compose([
            A.RandomScale(scale_limit=0.2),  # +/- 20%
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5,
                                 sat_shift_limit=30,
                                 val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3), rotate=(-40, 40), shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5,
                                 sat_shift_limit=30,
                                 val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale05_1':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.5, 1.0),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5,
                                 sat_shift_limit=30,
                                 val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_12':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.3, 1.2),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1),
                       p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5,
                                 sat_shift_limit=30,
                                 val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_scale03_07':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(
                scale=(0.3, 0.7),  # scale 512 to 256 in average
                rotate=(-40, 40),
                shear=(-0.1, 0.1),
                p=1),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5,
                                 sat_shift_limit=30,
                                 val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'distortions_light':
        transform = A.Compose([
            IAAPerspective2(scale=(0.0, 0.02)),
            IAAAffine2(scale=(0.8, 1.8), rotate=(-20, 20), shear=(-0.03, 0.03)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5,
                                 sat_shift_limit=30,
                                 val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'non_space_transform':
        transform = A.Compose([
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2,
                                       contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5,
                                 sat_shift_limit=30,
                                 val_shift_limit=5),
            A.ToFloat()
        ])
    elif transform_variant == 'no_augs':
        transform = A.Compose([A.ToFloat()])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform


def make_default_train_dataloader(indir,
                                  kind='default',
                                  out_size=512,
                                  mask_gen_kwargs=None,
                                  transform_variant='default',
                                  mask_generator_kind="mixed",
                                  dataloader_kwargs=None,
                                  ddp_kwargs=None,
                                  **kwargs):
    LOGGER.info(
        f'Make train dataloader {kind} from {indir}. Using mask generator={mask_generator_kind}'
    )

    mask_generator = get_mask_generator(kind=mask_generator_kind,
                                        kwargs=mask_gen_kwargs)
    transform = get_transforms(transform_variant, out_size)

    if kind == 'default':
        #dataset = InpaintingTrainDataset(indir=indir,
        #                                 mask_generator=mask_generator,
        #                                 transform=transform,
        #                                 **kwargs)
        dataset = DecaptioningTrainDataset('/ssd2/lixin41/LR540pcrf15.list',
                                           mask_generator=mask_generator)
    elif kind == 'default_web':
        dataset = InpaintingTrainWebDataset(indir=indir,
                                            mask_generator=mask_generator,
                                            transform=transform,
                                            **kwargs)
    elif kind == 'img_with_segm':
        dataset = ImgSegmentationDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         out_size=out_size,
                                         **kwargs)
    else:
        raise ValueError(f'Unknown train dataset kind {kind}')

    if dataloader_kwargs is None:
        dataloader_kwargs = {}

    is_dataset_only_iterable = kind in ('default_web', )

    if ddp_kwargs is not None and not is_dataset_only_iterable:
        dataloader_kwargs['shuffle'] = False
        dataloader_kwargs['sampler'] = DistributedSampler(dataset, **ddp_kwargs)

    if is_dataset_only_iterable and 'shuffle' in dataloader_kwargs:
        with open_dict(dataloader_kwargs):
            del dataloader_kwargs['shuffle']

    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_default_val_dataset(indir,
                             kind='default',
                             out_size=512,
                             transform_variant='default',
                             **kwargs):
    if OmegaConf.is_list(indir) or isinstance(indir, (tuple, list)):
        return ConcatDataset([
            make_default_val_dataset(idir,
                                     kind=kind,
                                     out_size=out_size,
                                     transform_variant=transform_variant,
                                     **kwargs) for idir in indir
        ])

    LOGGER.info(f'Make val dataloader {kind} from {indir}')
    mask_generator = get_mask_generator(kind=kwargs.get("mask_generator_kind"),
                                        kwargs=kwargs.get("mask_gen_kwargs"))

    if transform_variant is not None:
        transform = get_transforms(transform_variant, out_size)

    if kind == 'default':
        #dataset = InpaintingEvaluationDataset(indir, **kwargs)
        dataset = InferEvaluationDataset(
            mask_dir='/ssd3/lixin41/lama-main/LaMa_test_images/mask',
            image_dir='/ssd3/lixin41/lama-main/LaMa_test_images/frames')
    elif kind == 'our_eval':
        dataset = OurInpaintingEvaluationDataset(indir, **kwargs)
    elif kind == 'img_with_segm':
        dataset = ImgSegmentationDataset(indir=indir,
                                         mask_generator=mask_generator,
                                         transform=transform,
                                         out_size=out_size,
                                         **kwargs)
    elif kind == 'online':
        dataset = InpaintingEvalOnlineDataset(indir=indir,
                                              mask_generator=mask_generator,
                                              transform=transform,
                                              out_size=out_size,
                                              **kwargs)
    else:
        raise ValueError(f'Unknown val dataset kind {kind}')

    return dataset


def make_default_val_dataloader(*args, dataloader_kwargs=None, **kwargs):
    dataset = make_default_val_dataset(*args, **kwargs)

    if dataloader_kwargs is None:
        dataloader_kwargs = {}
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    return dataloader


def make_constant_area_crop_params(img_height,
                                   img_width,
                                   min_size=128,
                                   max_size=512,
                                   area=256 * 256,
                                   round_to_mod=16):
    min_size = min(img_height, img_width, min_size)
    max_size = min(img_height, img_width, max_size)
    if random.random() < 0.5:
        out_height = min(
            max_size,
            ceil_modulo(random.randint(min_size, max_size), round_to_mod))
        out_width = min(max_size, ceil_modulo(area // out_height, round_to_mod))
    else:
        out_width = min(
            max_size,
            ceil_modulo(random.randint(min_size, max_size), round_to_mod))
        out_height = min(max_size, ceil_modulo(area // out_width, round_to_mod))

    start_y = random.randint(0, img_height - out_height)
    start_x = random.randint(0, img_width - out_width)
    return (start_y, start_x, out_height, out_width)
