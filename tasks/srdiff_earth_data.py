import torchvision.transforms as transforms
from PIL import Image

from tasks.srdiff import SRDiffTrainer
from utils.dataset import DualSourceSRDataSet
from utils.hparams import hparams
from utils.indexed_datasets import IndexedDataset


class EarthDataSet(DualSourceSRDataSet):
    def __init__(self, prefix='train'):
        super().__init__(prefix)
        preprocess_transforms = []
        if self.prefix == 'test':
            self.len = 5000
            if hparams['test_save_png']:
                self.test_ids = hparams['test_ids']
                self.len = len(self.test_ids)

    def _get_item(self, index):
        if self.indexed_ds is None:
            self.indexed_ds = IndexedDataset(f'{self.data_dir}/{self.prefix}')
        if self.prefix == 'test' and hparams['test_save_png']:
            return self.indexed_ds[self.test_ids[index]]
        else:
            return self.indexed_ds[index]



class SRDiffEarthData(SRDiffTrainer):
    def __init__(self):
        super().__init__()
        self.dataset_cls = EarthDataSet
