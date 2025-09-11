import torch
import numpy as np
from cupy import shape

from configure import Config
from model import WNet
from Ncuts import NCutsLoss
from DataLoader import DataLoader
import time
import os
import torchvision
import pdb
from PIL import Image
import os

# os.environ["CUDA_VISIBLE_DEVICES"]=config.cuda_dev_list

if __name__ == '__main__':
    for dataset_type in ['bsd300','bsd500']:
        bsds = r"D:\DataSet\BSD\300\BSDS300\images" if dataset_type=='bsd300' else r'D:\DataSet\BSD\500\BSDS500\data\images'
        checkpoints_main_dir = fr"D:\Study\runs\competetors\WNET\checkpoints\{dataset_type}"
        main_output_dir = fr'D:\Study\runs\competetors\WNET\labs\{dataset_type}'
        for K in range(2, 7):

            checkpoints_dir = os.path.join(checkpoints_main_dir,str(K))
            output_dir = os.path.join(main_output_dir,str(K))
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)
            config = Config(K,dataset_type,bsds=bsds)
            config.model_tested=''
            for file in os.listdir(checkpoints_dir):
                if '_epoch_0200' in file:
                    config.model_tested = os.path.join(checkpoints_dir, file)

            # dataset_type = 'drive'

            dir_ = os.path.join('.','images',dataset_type)
            if not os.path.isdir(dir_):
                os.makedirs(dir_)
            dataset = DataLoader(config.bsds,"test",config)
            dataloader = dataset.torch_loader(shuffle=False)
            model = WNet(config)
            model.cuda()
            model.eval()
            optimizer = torch.optim.SGD(model.parameters(),lr = config.init_lr)
            #optimizer
            with open(config.model_tested,'rb') as f:
                para = torch.load(f,"cuda:0")
                model.load_state_dict(para['state_dict'])

            count = 0
            for step,[x] in enumerate(dataloader):
                print('Step' + str(step+1))
                    #NCuts Loss
                batch_im_names = dataset.im_names[count:count + len(x)]
                batch_im_sizes = dataset.im_sizes[count:count + len(x)]

                count += len(x)
                x = x.cuda()
                pred,pad_pred = model(x)
                seg = (pred.argmax(dim = 1)).cpu().detach().numpy()
                x = x.cpu().detach().numpy()*255
                x = np.transpose(x.astype(np.uint8),(0,2,3,1))
                color_map = lambda c: config.color_lib[c]
                cmap = np.vectorize(color_map)
                seg = np.moveaxis(np.array(cmap(seg)),0,-1).astype(np.uint8)
                #pdb.set_trace()
                for i,im_name,im_size in zip(range(seg.shape[0]),batch_im_names,batch_im_sizes):
                    # Image.fromarray(x[i]).save(f"./images/{dataset_type}/input_"+str(step+1)+"_"+str(i)+".jpg")
                    #for j in range(seg.shape[-1]):
                    #pdb.set_trace()
                    # Image.fromarray(seg[i,:,:]).save(f"./images/{dataset_type}/seg_"+str(step+1)+"_"+str(i)+".png")
                    Image.fromarray(seg[i, :, :]).resize(im_size, Image.NEAREST).save(os.path.join(output_dir,f'{im_name}.png'))





        
