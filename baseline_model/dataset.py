from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os



class TripletData(Dataset):
    def __init__(self, pairs, transform, img_paths, set):
        self.pairs = pairs
        self.transform = transform
        self.img_paths = img_paths
        self.set = set

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        user_img = Image.open(os.path.join(self.set, self.set, 'cropped',self.img_paths[self.pairs[i]['p'][0]]))
        shop_p_img = Image.open(os.path.join(self.set, self.set, 'cropped',self.img_paths[self.pairs[i]['p'][1]]))
        shop_n_img = Image.open(os.path.join(self.set, self.set, 'cropped',self.img_paths[self.pairs[i]['n'][1]]))


        if self.transform is not None:
            user_img = self.transform(user_img)
            shop_p_img = self.transform(shop_p_img)
            shop_n_img = self.transform(shop_n_img)

        return user_img, shop_p_img, shop_n_img


class RetrievalData(Dataset):
    def __init__(self, idx_set , pairs, transform, img_paths, set):
        self.idx_set = idx_set
        self.pairs = pairs
        self.transform = transform
        self.img_paths = img_paths
        self.set = set

    def __len__(self):
        return len(self.idx_set)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.set, self.set, 'cropped', self.img_paths[self.idx_set[i]]))
        if self.transform is not None:
            img = self.transform(img)

        return img
