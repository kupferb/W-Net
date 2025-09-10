import platform

class Config:
    
    def __init__(self,K=2):
        #network configure
        self.InputCh=3
        self.ScaleRatio = 2
        self.ConvSize = 3
        self.pad = 1#(self.ConvSize - 1) / 2 
        self.MaxLv = 5
        self.ChNum = [self.InputCh,64]
        for i in range(self.MaxLv-1):
            self.ChNum.append(self.ChNum[-1]*2)
        #data configure
        if platform.system().lower() == 'windows':
            self.pascal = r"D:\DataSet\PASCAL\VOCdevkit\VOC2012"
            self.pascal = r"D:\DataSet\PASCAL\VOCdevkit\VOC2012"
            self.pascal = r'C:\DataSet\DRIVE\training\images_wnet'
            self.pascal = r"D:\DataSet\BSD\300\BSDS300\images"
        else:
            self.pascal = r"/home/benk/Study/VOC2012"
            self.pascal = r"/home/benk/Downloads/BSDS300-images/BSDS300/images"
        #test
        self.bsds = r"D:\DataSet\STARE\test\images_wnet"
        self.bsds = r"D:\DataSet\DRIVE\test\images_wnet"
        self.bsds = r"D:\DataSet\BSD\300\BSDS300\images"
        self.imagelist = "ImageSets/Segmentation/train.txt"
        self.imagelist = "iids_train.txt"

        self.BatchSize = 10
        self.Shuffle = True
        self.LoadThread = 16
        self.inputsize = [224,224]
        #partition configure
        self.K = K
        #training configure
        self.epochs_to_save = 50
        self.init_lr = 0.05
        self.lr_decay = 0.1
        self.lr_decay_iter = 1000
        self.max_iter = 200
        self.cuda_dev = 0 
        self.cuda_dev_list = "0"#"0,1"
        self.check_iter = 1000
        #Ncuts Loss configure
        self.radius = 4
        self.sigmaI = 10
        self.sigmaX = 4
        #testing configure
        self.model_tested = "./checkpoints/checkpoints_bsd300/checkpoint_6_10_16_10_epoch_900"
        self.model_tested = r".\checkpoints\bsd300\2\checkpoint_9_10_16_46_epoch_0001"
        #color library
        self.color_lib = []
        self.color_lib = [(0,0,0),(255,255,255)]
        # for r in range(0,256,64):
        #     for g in range(0,256,64):
        #         for b in range(0,256,64):
        #             self.color_lib.append((r,g,b))
