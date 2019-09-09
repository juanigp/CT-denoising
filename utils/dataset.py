import pandas as pd
import torch
import torch.utils.data


class CTVolumesDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file):
        self.dataframe = pd.read_csv(csv_file)
        #self.root_dir = root_dir

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        #access the dataframe at the row idx, at columns 0 and 1 respectively
        lo_res_dir = self.dataframe.iloc[idx, 0]
        hi_res_dir = self.dataframe.iloc[idx, 1]
        
        #load the patches
        lo_res_patch = torch.load(lo_res_dir)
        hi_res_patch = torch.load(hi_res_dir)

        #tuple with the pair
        pair = (lo_res_patch, hi_res_patch)
        return pair


def var_or_cuda(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)        