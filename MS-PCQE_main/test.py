

# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import scipy
from scipy import stats
import scipy.io as scio
from scipy.optimize import curve_fit
from data_loader import VideoDataset_NR_image_with_fast_features
import ResNet_mean_with_fast
import time
from tqdm import tqdm
import torch.nn.functional as F





def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
    logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
    yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
    return yhat

def fit_function(y_label, y_output):
    beta = [np.max(y_label), np.min(y_label), np.mean(y_output), 0.5]
    popt, _ = curve_fit(logistic_func, y_output, \
        y_label, p0=beta, maxfev=100000000)
    y_output_logistic = logistic_func(y_output, *popt)
    
    return y_output_logistic




def main(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_all = np.zeros([config.split_num, 4])
    final_all = np.zeros([config.split_num, 4])
    for split in range(config.split_num):
        # model
        if config.model_name == 'ResNet_mean_with_fast':
            print('The current model is ' + config.model_name)
            model = ResNet_mean_with_fast.resnet18(pretrained=True)
        if config.multi_gpu:
            model = torch.nn.DataParallel(model, device_ids=[int(id) for id in config.gpu_ids])
            model = model.to(device)
        else:
            model = model.to(device)




        param_num = 0
        for param in model.parameters():
            param_num += int(np.prod(param.shape))
        print('Trainable params: %.2f million' % (param_num / 1e6))
        

        print('*******************************************************************************************************************************************************')
        print('Using '+ str(split+1) + '-th split.' )

        transformations_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])


        transformations_test = transforms.Compose([transforms.ToTensor(),  transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])


        transformations_mask = transforms.Compose([transforms.ToTensor()])





        if config.database == 'SJTU':

            images_dir_10 = '../rotation/sjtu_scale_0.75_512_12/frames_sjtu'
            images_dir_05 = '../rotation/sjtu_scale_0.5_512_12/frames_sjtu'

            datainfo_train = 'database/sjtu_data_info/train_' + str(split + 1) + '.csv'
            datainfo_test = 'database/sjtu_data_info/test_' + str(split + 1) + '.csv'

            trainset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                                images_dir_10, images_dir_05, datainfo_train,
                                                                transformations_train, crop_size=config.crop_size,
                                                                mode_idx=0)

            testset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                               images_dir_10, images_dir_05, datainfo_test,
                                                               transformations_test, crop_size=config.crop_size,
                                                               mode_idx=1)


        elif config.database == 'WPC':
            images_dir_10 = '../rotation/wpc_scale_0.75_512_12/frames_wpc'
            images_dir_05 = '../rotation/wpc_scale_0.5_512_1/frames_wpc'



            datainfo_train = 'database/wpc_data_info/train_'+ str(split+1) +'.csv'
            datainfo_test = 'database/wpc_data_info/test_'+ str(split+1) +'.csv'

            trainset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                                images_dir_10, images_dir_05,datainfo_train,
                                                                transformations_train, crop_size=config.crop_size, mode_idx=0)

            testset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                               images_dir_10, images_dir_05, datainfo_test,
                                                               transformations_test, crop_size=config.crop_size, mode_idx=1)


        elif config.database == 'IRPC':
            images_dir_10 = '../rotation/irpc_scale_0.75_512_12/frames_irpc'
            images_dir_05 = '../rotation/irpc_scale_0.5_512_12/frames_irpc'

            datainfo_train = 'database/irpc_data_info/train_' + str(split + 1) + '.csv'
            datainfo_test = 'database/irpc_data_info/test_' + str(split + 1) + '.csv'

            trainset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                                images_dir_10, images_dir_05, datainfo_train,
                                                                transformations_train, crop_size=config.crop_size,
                                                                mode_idx=0)

            testset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                               images_dir_10, images_dir_05, datainfo_test,
                                                               transformations_test, crop_size=config.crop_size,
                                                               mode_idx=1)


        elif config.database == 'MPCCD':
            images_dir_10 = '../rotation/mpccd_scale_0.75_512_12/frames_mpccd'
            images_dir_05 = '../rotation/mpccd_scale_0.5_512_12/frames_mpccd'

            datainfo_train = 'database/mpccd_data_info/train_' + str(split + 1) + '.csv'
            datainfo_test = 'database/mpccd_data_info/test_' + str(split + 1) + '.csv'

            trainset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                                images_dir_10, images_dir_05, datainfo_train,
                                                                transformations_train, crop_size=config.crop_size,
                                                                mode_idx=0)

            testset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                               images_dir_10, images_dir_05, datainfo_test,
                                                               transformations_test, crop_size=config.crop_size,
                                                               mode_idx=1)

        elif config.database == 'SIAT':
            images_dir_10 = '../rotation/siat_scale_0.75_512_12/frames_siat'
            images_dir_05 = '../rotation/siat_scale_0.5_512_12/frames_siat'

            datainfo_train = 'database/siat_data_info/train_' + str(split + 1) + '.csv'
            datainfo_test = 'database/siat_data_info/test_' + str(split + 1) + '.csv'

            trainset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                                images_dir_10, images_dir_05, datainfo_train,
                                                                transformations_train, crop_size=config.crop_size,
                                                                mode_idx=0)

            testset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                               images_dir_10, images_dir_05, datainfo_test,
                                                               transformations_test, crop_size=config.crop_size,
                                                               mode_idx=1)


        elif config.database == 'LSPCQA':
            images_dir_10 = 'rotation/lspcqa_scale_0.75_512_12/frames_lspcqa'
            images_dir_05 = 'rotation/lspcqa_scale_0.5_512_12/frames_lspcqa'

            datainfo_train = 'database/lspcqa_data_info/train_' + str(split + 1) + '.csv'
            datainfo_test = 'database/lspcqa_data_info/test_' + str(split + 1) + '.csv'

            trainset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                                images_dir_10, images_dir_05, datainfo_train,
                                                                transformations_train, crop_size=config.crop_size,
                                                                mode_idx=0)

            testset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                               images_dir_10, images_dir_05, datainfo_test,
                                                               transformations_test, crop_size=config.crop_size,
                                                               mode_idx=1)
        elif config.database == 'REHEARSE':
            images_dir_10 = '../alpha_0.005_frames'
            images_dir_05 = '../alpha_0.005_frames'
            datainfo_train = 'database/lspcqa_data_info/train_' + str(split + 1) + '.csv'
            datainfo_test = 'database/lspcqa_data_info/test_' + str(split + 1) + '.csv'


            testset = VideoDataset_NR_image_with_fast_features(transformations_mask, config.sampling_div,
                                                    images_dir_10, images_dir_05, datainfo_test,
                                                    transformations_test, crop_size=config.crop_size,
                                                    mode_idx=1)



        ## dataloader
        test_loader = torch.utils.data.DataLoader(testset, batch_size=1,
            shuffle=False, num_workers=config.num_workers)


        best_test_criterion = -1  # SROCC min
        final = np.zeros(4)
        best = np.zeros(4)
        n_test = len(testset)
        print('Starting training:')
        #model.load_state_dict(torch.load('ckpts/' + config.database + '/ResNet_mean_with_fast_' + config.database + '_' + str(split+1) + '_' + 'best.pth'))  # SJTU/WPC
        model.load_state_dict(torch.load('ckpts/' + 'LSPCQA' + '/ResNet_mean_with_fast_' + 'LSPCQA' + '_' + str(split+1) + '_' + 'best.pth'))

        # Test
        model.eval()
        y_output = np.zeros(n_test)
        y_test = np.zeros(n_test)

        tqdm_test = tqdm(test_loader, ncols=80)


        # do validation after each epoch
        with torch.no_grad():
            for i, (frames_dir_10_video, frames_dir_05_video, frames_dir_10_video_mask, frames_dir_05_video_mask, labels) in enumerate(tqdm_test):
                y_test[i] = labels.item()

                y_mid_ref = model(frames_dir_10_video, frames_dir_05_video, frames_dir_10_video_mask,frames_dir_05_video_mask, mode_idx=0)
                y_output[i] = y_mid_ref.item()



            y_output_logistic = fit_function(y_test, y_output)
            test_PLCC = stats.pearsonr(y_output_logistic, y_test)[0]
            test_SROCC = stats.spearmanr(y_output, y_test)[0]
            test_RMSE = np.sqrt(((y_output_logistic-y_test) ** 2).mean())
            test_KROCC = scipy.stats.kendalltau(y_output, y_test)[0]
            print("Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(test_SROCC, test_KROCC, test_PLCC, test_RMSE))
            final[0:4] = [test_SROCC, test_KROCC, test_PLCC, test_RMSE]
            final_all[split, :] = final

            if test_SROCC > best_test_criterion:
                # scio.savemat('ckpts/LSPCQA/y_output' + str(split) + '.mat', {'y_output': y_output})
                # scio.savemat('ckpts/LSPCQA/y_label' + str(split) + '.mat', {'y_label': y_test})
                best[0:4] = [test_SROCC, test_KROCC, test_PLCC, test_RMSE]
                best_all[split, :] = best


        print('*************************************************************************************************************************')
        print("The best Test results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(best[0], best[1], best[2], best[3]))



    final_mean = np.mean(best_all, 0)
    # scio.savemat('ckpts/LSPCQA/best_all.mat', {'best_all': best_all})
    print('*************************************************************************************************************************')
    print("The mean val results: SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}".format(final_mean[0], final_mean[1], final_mean[2], final_mean[3]))
    print('*************************************************************************************************************************')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # input parameters
    parser.add_argument('--database', type=str, default='LSPCQA')
    parser.add_argument('--model_name', type=str, default='ResNet_mean_with_fast')

    # training parameters
    parser.add_argument('--conv_base_lr', type=float, default=5e-5)
    parser.add_argument('--decay_ratio', type=float, default=0.5)
    parser.add_argument('--decay_interval', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)  # epochs = 40 实际epoch设置为40，数据集index扩大5倍，即每5次做一次数据测试与模型保存（好处是数据载入更快）
    parser.add_argument('--split_num', type=int, default=5)
    parser.add_argument('--crop_size', type=int, default=512) # 512
    parser.add_argument('--sampling_div', type=int, default=6)


    # misc
    parser.add_argument('--ckpt_path', type=str, default='ckpts')
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu_ids', type=list, default=None)

    config = parser.parse_args()

    main(config)


