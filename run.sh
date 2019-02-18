# #!/usr/bin/env bash


#THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32  python train.py --resume HREDtwitter480wan_d_wo/1524532070.36_TwitterModel --save_every_valid_iteration --prototype prototype_twitter_HRED > Model_Output_nlpcc_hred_480wan6wan_w_0_d_2

#THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32  python train.py --resume HREDtwitter480wan_d_wo/1524532070.36_TwitterModel --save_every_valid_iteration --prototype prototype_twitter_HRED > Model_Output_nlpcc_hred_480wan6wan_w_0_d_3

#THEANO_FLAGS=mode=FAST_RUN,device=gpu1,floatX=float32  python train.py --resume HREDtwitter480wan_d_wo/1524532070.36_TwitterModel --save_every_valid_iteration --prototype prototype_twitter_HRED > Model_Output_nlpcc_hred_480wan6wan_w_0_d_4

THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32  python train.py --save_every_valid_iteration --prototype prototype_ubuntu_HRED_kg > Model_Output_kg

#THEANO_FLAGS=mode=FAST_RUN,device=gpu7,floatX=float32 python train.py --resume menu1.8wanci/1543999622.11_TwitterModel --save_every_valid_iteration --prototype prototype_ubuntu_HRED_kg > Model_Output_kg3THEANO_FLAGS=mode=FAST_RUN,device=cpu,floatX=float32  python train.py --save_every_valid_iteration --prototype prototype_ubuntu_HRED_kg > Model_Output_kg_cpu
#THEANO_FLAGS=device=gpu7,floatX=float32,optimizer=None  python train.py --save_every_valid_iteration --prototype prototype_ubuntu_HRED_kg > Model_Output_kg
