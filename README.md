# Improved-ERFNet
erfnet.py                     ERFNet网络构建
erfnet_pspnet.py              ERF-PSPNet网络构建
erfnet_sepspnet.py            ERF-SEPSPNet网络构建
erfnet_psp_senet.py           ERF-PSPSENet网络构建
erfnet_sepspnet_contactSE.py  ERF-SEPSPNet_contactSE网络构建
erfnet_pspsenet_contactSE.py  ERF-PSPSENet_contactSE网络构建

五个改进ERFNet模型训练、测试过程：
#############################################################################################################################3
ERFNet的运行过程：
训练：python main_erfnet.py --savedir erfnet_training1 
编码器阶段：
python main_erfnet.py --savedir erfnet_training1 --datadir /media/a104/D/stli/data/cityscapes/ --num-epochs 150 --batch-size 6 
解码器阶段：
python main_erfnet.py --savedir erfnet_training1 --datadir /media/a104/D/stli/data/cityscapes/ --num-epochs 150 --batch-size 6 --decoder --pretrainedEncoder "/media/a104/D/stli/erfnet/save/erfnet_training1/model_best_enc.pth.tar"
测试：
多张图片的可视化结果：
python infer_erfnet.py

###############################################################################################################################
ERF_PSPNet的运行过程:
训练： python main_pspnet.py --savedir erfnet_training2
编码器阶段：
python main_pspnet.py --savedir erfnet_training2 --datadir /media/a104/D/stli/data/cityscapes/ --num-epochs 150 --batch-size 6
解码器阶段：
python main_pspnet.py --savedir erfnet_training2 --datadir /media/a104/D/stli/data/citysca pes/ --num-epochs 150 --batch-size 6 --decoder --pretrainedEncoder "/media/a104/D/stli/erfnet/save/erfnet_training2/model_best_enc.pth.tar"
python main_pspnet_classiou.py --savedir erfnet_training2_classiou --datadir /media/a104/D/stli/data/cityscapes/ --num-epochs 150 --batch-size 6 --decoder --pretrainedEncoder "/media/a104/D/stli/erfnet/save/erfnet_training2/model_best_enc.pth.tar"
测试：
多张图片的可视化结果：
python infer_erfnet_pspnet.py

###################################################################################################################################
ERF_SEPSPNet的运行过程：
训练：python main_sepspnet.py --savedir erfnet_training3
编码器阶段：
python main_sepspnet.py --savedir erfnet_training3 --datadir /media/a104/D/stli/data/cityscapes/ --num-epochs 150 --batch-size 6
解码器阶段：
python main_sepspnet.py --savedir erfnet_training3 --datadir /media/a104/D/stli/data/cityscapes/ --num-epochs 150 --batch-size 6 --decoder --pretrainedEncoder "/media/a104/D/stli/erfnet/save/erfnet_training3/model_best_enc.pth.tar"
python main_sepspnet_classiou.py --savedir erfnet_training3_classiou --datadir /media/a104/D/stli/data/cityscapes/ --num-epochs 150 --batch-size 6 --decoder --pretrainedEncoder "/media/a104/D/stli/erfnet/save/erfnet_training3/model_best_enc.pth.tar"
测试：
多张图片的可视化结果：
python infer_erfnet_sepspnet.py

#####################################################################################################################################3
ERF_PSP_SENet的运行过程：
训练：python main_psp_senet.py --savedir erfnet_training4
python main_psp_senet_classiou.py --savedir erfnet_training4 --datadir /media/a104/D/stli/data/cityscapes/ --num-epochs 150 --batch-size 6 --decoder --pretrainedEncoder "/media/a104/D/stli/erfnet/save/erfnet_training4/model_best_enc.pth.tar"

########################################################################################################################################3
ERF_SEPSPNet_contactSE的运行过程 2>&1 | tee erfnet_sepspnet_contactSE_flops.log
训练：CUDA_VISIBLE_DEVICES=0,1 python main_sepspnet_contactSE_classiou.py --savedir erfnet_training_sepspnet_contactSE
CUDA_VISIBLE_DEVICES=0,1 python main_sepspnet_contactSE_classiou.py --savedir erfnet_training_sepspnet_contactSE --datadir /media/a104/D/stli/data/cityscapes/ --num-epochs 200 --batch-size 6 --decoder --pretrainedEncoder "/media/a104/D/stli/erfnet/save/erfnet_training_sepspnet_contactSE/model_best_enc.pth.tar"
测试：python infer_erfnet_sepspnet_contactSE.py

####################################################################################################################################3
ERFNet_PSPSENet_contactSE的运行过程
训练：CUDA_VISIBLE_DEVICES=0,1 python main_pspsenet_contactSE_classiou.py --savedir erfnet_training_pspsenet_contactSE
测试：python infer_erfnet_pspsenet_contactSE.py
