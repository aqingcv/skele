import numpy as np
import pickle, torch
from . import tools


# class Feeder_single(torch.utils.data.Dataset):
#     """ Feeder for single inputs """

#     def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, rotate = 1,mmap=True):
#         self.data_path = data_path
#         self.label_path = label_path

#         self.shear_amplitude = shear_amplitude
#         self.temperal_padding_ratio = temperal_padding_ratio
#         self.rotate = rotate
       
#         self.load_data(mmap)

#     def load_data(self, mmap):
#         # load label
#         with open(self.label_path, 'rb') as f:
#             self.sample_name, self.label = pickle.load(f)

#         # load data
#         if mmap:
#             self.data = np.load(self.data_path, mmap_mode='r')
#         else:
#             self.data = np.load(self.data_path)

#     def __len__(self):
#         return len(self.label)

#     def __getitem__(self, index):
#         # get data
#         data_numpy = np.array(self.data[index])
#         label = self.label[index]
        
#         # processing
#         data = self._aug(data_numpy)
#         return data, label

#     def _aug(self, data_numpy):
#         if self.temperal_padding_ratio > 0:
#             data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

#         if self.shear_amplitude > 0:
#             data_numpy = tools.shear(data_numpy, self.shear_amplitude)

#         if self.rotate > 0 :    
#             data_numpy = tools.random_rotate(data_numpy)  

#         return data_numpy

class Feeder_single(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, rotate=1, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.rotate = rotate
        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # processing
        data = self._aug(data_numpy)
        return data, label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        
        if self.rotate > 0:
            data_numpy = tools.random_rotate(data_numpy)   

        return data_numpy
        
class Feeder_dual(torch.utils.data.Dataset):
    """ Feeder for dual inputs """

    def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, rotate=1, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.rotate = rotate
        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # processing
        data1 = self._aug(data_numpy)
        data2 = self._aug(data_numpy)
        return [data1, data2], label

    def _aug(self, data_numpy):
        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        if self.rotate > 0:
            data_numpy = tools.random_rotate(data_numpy)   
                 
        return data_numpy

class Feeder_triple(torch.utils.data.Dataset):
    """ Feeder for dual inputs """

    def __init__(self, data_path, label_path, num_frame_path, l_ratio, input_size, 
                 shear_amplitude=0.5, temperal_padding_ratio=6, rotate=1, mmap=True):
                 
        self.data_path = data_path
        self.label_path = label_path
        self.num_frame_path= num_frame_path
        self.input_size=input_size
        self.l_ratio = l_ratio
        self.shear_amplitude = shear_amplitude
        self.temperal_padding_ratio = temperal_padding_ratio
        self.rotate = rotate

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        self.number_of_frames= np.load(self.num_frame_path)


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        number_of_frames = self.number_of_frames[index]
        label = self.label[index]

        data1_crop = tools.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)
        data2_crop = tools.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)

        # processing
        data1 = self._aug(data1_crop)
        data2 = self._aug(data2_crop)

        return [data1, data2], label


    def _aug(self, data_numpy):

        if self.temperal_padding_ratio > 0:
            data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)

        if self.rotate > 0 :    
            data_numpy = tools.random_rotate(data_numpy)

        data_numpy = tools.random_spatial_flip(data_numpy)
        
        return data_numpy

    # def _strong_aug(self, data_numpy):
    #     if self.temperal_padding_ratio > 0:
    #         data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)
    #     if self.shear_amplitude > 0:
    #         data_numpy = tools.shear(data_numpy, self.shear_amplitude)

    #     data_numpy = tools.random_rotate(data_numpy)
    #     data_numpy = tools.gaus_noise(data_numpy)

    #     return data_numpy



# class Feeder_semi(torch.utils.data.Dataset):
#     """ Feeder for semi-supervised learning """

#     def __init__(self, data_path, label_path, shear_amplitude=0.5, temperal_padding_ratio=6, mmap=True, label_list=None):
#         self.data_path = data_path
#         self.label_path = label_path

#         self.shear_amplitude = shear_amplitude
#         self.temperal_padding_ratio = temperal_padding_ratio
#         self.label_list = label_list
       
#         self.load_data(mmap)
#         self.load_semi_data()    

#     def load_data(self, mmap):
#         # load label
#         with open(self.label_path, 'rb') as f:
#             self.sample_name, self.label = pickle.load(f)

#         # load data
#         if mmap:
#             self.data = np.load(self.data_path, mmap_mode='r')
#         else:
#             self.data = np.load(self.data_path)

#     def load_semi_data(self):
#         data_length = len(self.label)

#         if not self.label_list:
#             self.label_list = list(range(data_length))
#         else:
#             self.label_list = np.load(self.label_list).tolist()
#             self.label_list.sort()

#         self.unlabel_list = list(range(data_length))

#     def __len__(self):
#         return len(self.unlabel_list)

#     def __getitem__(self, index):
#         # get data
#         data_numpy = np.array(self.data[index])
#         label = self.label[index]
        
#         # processing
#         data = self._aug(data_numpy)
#         return data, label
    
#     def __getitem__(self, index):
#         label_index = self.label_list[index % len(self.label_list)]
#         unlabel_index = self.unlabel_list[index]

#         # get data
#         label_data_numpy = np.array(self.data[label_index])
#         unlabel_data_numpy = np.array(self.data[unlabel_index])
#         label = self.label[label_index]
        
#         # processing
#         data1 = self._aug(unlabel_data_numpy)
#         data2 = self._aug(unlabel_data_numpy)
#         return [data1, data2], label_data_numpy, label

#     def _aug(self, data_numpy):
#         if self.temperal_padding_ratio > 0:
#             data_numpy = tools.temperal_crop(data_numpy, self.temperal_padding_ratio)

#         if self.shear_amplitude > 0:
#             data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        
#         return data_numpy