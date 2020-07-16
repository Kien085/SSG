from __future__ import print_function
import argparse
import numpy as np

parser = argparse.ArgumentParser()
# Input paths
parser.add_argument('--id', type=str, default='0',
                help='id of image')
parser.add_argument('--mode', type=str,  default='sen',
                help='image or sen')

opt = parser.parse_args()
if opt.mode == 'sen':
    sg_dict = np.load('spice_sg_dict2.npz')['spice_dict'][()]
    sg_dict = sg_dict['ix_to_word']
    folder = 'coco_spice_sg2/'
else:
    sg_dict = np.load('coco_pred_sg_rela.npy')[()]
    sg_dict = sg_dict['i2w']
    folder = 'coco_img_sg/'

sg_path = folder + opt.id + '.npy'
sg_use = np.load(sg_path)[()]
if opt.mode == 'sen':
    rela = sg_use['rela_info']
    obj_attr = sg_use['obj_info']
else:
    rela = sg_use['rela_matrix']
    obj_attr = sg_use['obj_attr']
N_rela = len(rela)
N_obj = len(obj_attr)
for i in range(N_obj):
    if opt.mode == 'sen':
        print('obj #{0}'.format(i), end = ': ')
        if len(obj_attr[i]) >= 2:
            print ('(', end = '')
            for j in range(len(obj_attr[i])-1):
                print('{0} '.format(sg_dict[obj_attr[i][j + 1]]), end = '')
            print (') ', end = '')
        print(sg_dict[obj_attr[i][0]])
    else:
        print('obj #{0}'.format(i), end = ': ')  # maybe it means 'bounding box' but not 'object'
        N_attr = 3
        for j in range(N_attr - 1):
            print('{0} {1}, '.format(sg_dict[obj_attr[i][j + 4]],\
                sg_dict[obj_attr[i][j+1]]), end = '')
        j = N_attr - 1
        print('{0} {1}'.format(sg_dict[obj_attr[i][j + 4]],\
            sg_dict[obj_attr[i][j+1]]))

for i in range(N_rela):
    obj_idx = 0 if opt.mode == 'sen' else 1
    sbj = sg_dict[ int(obj_attr[int(rela[i][0])][obj_idx]) ]
    obj = sg_dict[ int(obj_attr[int(rela[i][1])][obj_idx]) ]
    rela_name = sg_dict[rela[i][2]]
    print('rel #{3}: {0}-{1}-{2}'.format(sbj,rela_name,obj,i))