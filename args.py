
class args():

    # training args
    epochs = 100
    batch_size = 8
    trainNumber = 1600
    HEIGHT = 128
    WIDTH = 128
    in_c = 2
    out_c = 1

    Patch_path = "" # path to training image patches
    Patch_path2 = ""
    save_model = ""
    root_savemodel = ""
    cross_tag=0
    re_times=1
    cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"


    lr = 1e-4 #"learning rate, default is 1e-4"
    log_interval = 400 #"number of images after which the training loss is logged, default is 500"
    device = 0;


    #settings for test
    test_vi_path = ""
    test_ir_path = ""
    root_testoutput = ""
    pretrained_cnn_path = "pretrained_models/CNN/CNN.model"
    pretrained_trans_path = "pretrained_models/Trans/Trans.model"
    pretrained_gan_path = "pretrained_models/GAN/GAN.model"





