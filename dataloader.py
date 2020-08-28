from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
import os
import numpy as np
import random
from models.ass_fun import *
from functools import reduce

import torch
import torch.utils.data as data

import multiprocessing


class DataLoader(data.Dataset):

    def reset_iterator(self, split):
        del self._prefetch_process[split]
        self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
        self.iterators[split] = 0

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.ix_to_word

    def get_seq_length(self):
        return self.seq_length

    def __init__(self,opt):
        self.opt = opt
        self.batch_size = 50 #self.opt.batch_size
        self.seq_per_img = 5 #opt.seq_per_img
        self.use_att = True #getattr(opt, 'use_att', True)
        self.input_ssg_dir = self.opt.input_ssg_dir

        # load the json file which contains additional information about the dataset
        print('DataLoader loading json file:', self.opt.input_json)
        self.info = json.load(open(self.opt.input_json)) #self.opt.input_json

        print('using new dict')
        sg_dict_info = np.load(self.opt.sg_dict_path, allow_pickle=True)['spice_dict'][()]
        self.ix_to_word = sg_dict_info['ix_to_word']
        self.vocab_size = len(self.ix_to_word)
        print('vocab size is ', self.vocab_size)

        #Caption data, might not need, maybe will need
        print('DataLoader loading h5 file: ', self.opt.input_label_h5)
        self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core')

        # load in the sequence data
        seq_size = self.h5_label_file['labels'].shape
        print("seq_size:{0}".format(seq_size))
        self.seq_length = seq_size[1]
        print('max sequence length in data is', self.seq_length)
        # load the pointers in full to RAM (should be small enough)
        self.label_start_ix = self.h5_label_file['label_start_ix'][:]
        self.label_end_ix = self.h5_label_file['label_end_ix'][:]

        self.num_images = self.label_start_ix.shape[0]
        print('read %d image features' %(self.num_images))

        self.split_ix = {'train': [], 'val': [], 'test': [], 'train_sg': [], 'val_sg': [], 'test_sg': []}
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]
            if img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
                self.split_ix['train'].append(ix)
            elif img['split'] == 'test':
                self.split_ix['test'].append(ix)


        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

        self.iterators = {'train': 0, 'val': 0, 'test': 0}

        self._prefetch_process = {} # The three prefetch process
        for split in self.iterators.keys():
            self._prefetch_process[split] = BlobFetcher(split, self, split=='train')
            # Terminate the child process when the parent exists
        def cleanup():
            print('Terminating BlobFetcher')
            for split in self.iterators.keys():
                del self._prefetch_process[split]
        import atexit
        atexit.register(cleanup)

    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.h5_label_file['labels'][ixl, :self.seq_length]
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.h5_label_file['labels'][ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def get_batch(self, split, batch_size=None, seq_per_img=None):
        batch_size = batch_size or self.batch_size
        seq_per_img = seq_per_img or self.seq_per_img

        ssg_rela_batch = []
        ssg_obj_batch = []
        ssg_attr_batch = []

        label_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'int')
        mask_batch = np.zeros([batch_size * seq_per_img, self.seq_length + 2], dtype = 'float32')

        wrapped = False

        infos = []
        gts = []

        for i in range(batch_size):
            # fetch image
            tmp_ssg, ix, tmp_wrapped = self._prefetch_process[split].get() #I'm not exactly sure how this works, maybe a bit more after doing getitem

            ssg_rela_batch.append(tmp_ssg['ssg_rela_matrix'])
            ssg_attr_batch.append(tmp_ssg['ssg_attr'])
            ssg_obj_batch.append(tmp_ssg['ssg_obj'])



            label_batch[i * seq_per_img : (i + 1) * seq_per_img, 1 : self.seq_length + 1] = self.get_captions(ix, seq_per_img)

            if tmp_wrapped:
                wrapped = True

            # Used for reward evaluation
            gts.append(self.h5_label_file['labels'][self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
        
            # record associated info as well
            info_dict = {}
            info_dict['ix'] = ix
            info_dict['id'] = self.info['images'][ix]['id']
            info_dict['file_path'] = self.info['images'][ix]['file_path']
            infos.append(info_dict)

        data = {}
        max_rela_len = max([_.shape[0] for _ in ssg_rela_batch])
        data['ssg_rela_matrix'] = np.ones([len(batch_size) * seq_per_img, max_rela_len, 3]) * -1
        for i in range(len(ssg_rela_batch)):
            data['ssg_rela_matrix'][i*seq_per_img:(i+1)*seq_per_img,0:len(ssg_rela_batch[i]),:] = ssg_rela_batch[i]
        data['ssg_rela_masks'] = np.zeros(data['ssg_rela_matrix'].shape[:2], dtype='float32')
        for i in range(len(ssg_rela_batch)):
            data['ssg_rela_masks'][i * seq_per_img:(i + 1) * seq_per_img, :ssg_rela_batch[i].shape[0]] = 1

        max_obj_len = max([_.shape[0] for _ in ssg_obj_batch])
        data['ssg_obj'] = np.ones([len(batch_size) * seq_per_img, max_obj_len])*-1
        for i in range(len(ssg_obj_batch)):
            data['ssg_obj'][i * seq_per_img:(i+1)*seq_per_img,0:len(ssg_obj_batch[i])] = ssg_obj_batch[i]
        data['ssg_obj_masks'] = np.zeros(data['ssg_obj'].shape, dtype='float32')
        for i in range(len(ssg_obj_batch)):
            data['ssg_obj_masks'][i * seq_per_img:(i+1) * seq_per_img,:ssg_obj_batch[i].shape[0]] = 1

        max_attr_len = max([_.shape[1] for _ in ssg_attr_batch])
        data['ssg_attr'] = np.ones([len(batch_size) * seq_per_img, max_obj_len, max_attr_len])*-1
        for i in range(len(ssg_obj_batch)):
            data['ssg_attr'][i * seq_per_img:(i+1)*seq_per_img,0:len(ssg_obj_batch[i]),0:ssg_attr_batch[i].shape[1]] = \
                ssg_attr_batch[i]
        data['ssg_attr_masks'] = np.zeros(data['ssg_attr'].shape, dtype='float32')
        for i in range(len(ssg_attr_batch)):
            for j in range(len(ssg_attr_batch[i])):
                N_attr_temp = np.sum(ssg_attr_batch[i][j,:] >= 0)
                data['ssg_attr_masks'][i * seq_per_img: (i+1) * seq_per_img, j, 0:int(N_attr_temp)] = 1

        data['labels'] = np.vstack(label_batch)
        # generate mask
        nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))
        for ix, row in enumerate(mask_batch):
            row[:nonzeros[ix]] = 1
        data['masks'] = mask_batch

        data['gts'] = gts # all ground truth captions of each images
        data['bounds'] = {'it_pos_now': self.iterators[split], 'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
        data['infos'] = infos

        return data

    # It's not coherent to make DataLoader a subclass of Dataset, but essentially, we only need to implement the following to functions,
    # so that the torch.utils.data.DataLoader can load the data according the index.
    # However, it's minimum change to switch to pytorch data loading.
    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        ix = index #self.split_ix[index]
        ssg_data = {}
        ssg_data['ssg_rela_matrix'] = {}
        ssg_data['ssg_attr'] = {}
        ssg_data['ssg_obj'] = {}
        if self.use_att:
            #I'm not sure if I really need the use_att
            path_temp = os.path.join(self.input_ssg_dir, str(self.info['images'][ix]['id']) + '.npy')
            if os.path.isfile(path_temp):
                ssg_info = np.load(os.path.join(path_temp))
                ssg_rela_matrix = ssg_info[()]['rela_info']
                ssg_obj_att_info = ssg_info[()]['obj_info']

                len_obj = len(ssg_obj_att_info)
                ssg_obj = np.zeros([len_obj,])
                if len_obj == 0:
                    ssg_rela_matrix = np.zeros([0,3])
                    ssg_attr = np.zeros([0,1])
                    ssg_obj = np.zeros([0,])
                else:
                    max_attr_len = max([len(_) for _ in ssg_obj_att_info])
                    ssg_attr = np.ones([len_obj,max_attr_len-1])*-1
                    for i in range(len_obj):
                        ssg_obj[i]= ssg_obj_att_info[i][0]
                        for j in range(1,len(ssg_obj_att_info[i])):
                            ssg_attr[i,j-1] = ssg_obj_att_info[i][j]

                ssg_data = {}
                ssg_data['ssg_rela_matrix'] = ssg_rela_matrix
                ssg_data['ssg_attr'] = ssg_attr
                ssg_data['ssg_obj'] = ssg_obj
            else:
                ssg_data = {}
                ssg_data['ssg_rela_matrix'] = np.zeros([0,3])
                ssg_data['ssg_attr'] = np.zeros([0,1])
                ssg_data['ssg_obj'] = np.zeros([0,])
        else:
            att_feat = np.zeros((1,1,1))
        return (ssg_data,ix)

    def __len__(self):
        return len(self.info['images'])

class SubsetSampler(torch.utils.data.sampler.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.
    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class BlobFetcher():
    """Experimental class for prefetching blobs in a separate process."""
    def __init__(self, split, dataloader, if_shuffle=False):
        """
        db is a list of tuples containing: imcrop_name, caption, bbox_feat of gt box, imname
        """
        self.split = split
        self.dataloader = dataloader
        self.if_shuffle = if_shuffle

    # Add more in the queue
    def reset(self):
        """
        Two cases for this function to be triggered:
        1. not hasattr(self, 'split_loader'): Resume from previous training. Create the dataset given the saved split_ix and iterator
        2. wrapped: a new epoch, the split_ix and iterator have been updated in the get_minibatch_inds already.
        """
        # batch_size is 1, the merge is done in DataLoader class
        self.split_loader = iter(data.DataLoader(dataset=self.dataloader,
                                            batch_size=1,
                                            sampler=SubsetSampler(self.dataloader.split_ix[self.split][self.dataloader.iterators[self.split]:]),
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=4, # 4 is usually enough #KIENFIX?
                                            collate_fn=lambda x: x[0]))

    def _get_next_minibatch_inds(self):
        max_index = len(self.dataloader.split_ix[self.split])
        wrapped = False

        ri = self.dataloader.iterators[self.split]
        ix = self.dataloader.split_ix[self.split][ri]

        ri_next = ri + 1
        if ri_next >= max_index:
            ri_next = 0
            if self.if_shuffle:
                random.shuffle(self.dataloader.split_ix[self.split])
            wrapped = True
        self.dataloader.iterators[self.split] = ri_next

        return ix, wrapped
    
    def get(self):
        if not hasattr(self, 'split_loader'):
            self.reset()

        ix, wrapped = self._get_next_minibatch_inds()
        tmp = self.split_loader.next()
        if wrapped:
            self.reset()

        assert tmp[4] == ix, "ix not equal"

        return tmp + [wrapped]