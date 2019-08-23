# Paper: Deep Learning for Direct Hybrid Precoding in Millimeter Wave Massive MIMO Systems  https://arxiv.org/abs/1905.13212
import numpy as np
import scipy.io as scio

from Model.conv1D_precoder import ConvPrecoder
from Model.importdata import read_data_sets
from keras import backend as K

Flag_HDF5file = True

def ParseArgs(args=None):
    import argparse
    parser =  argparse.ArgumentParser(
        description = 'Train and test for direct hybrid precoding')
    parser.add_argument('-train', '--IsTrain', nargs = '?', type=int, default=1,
                        help="""Set to 1 for training, 0 for testing""")
    return parser.parse_args(args)

if __name__ == '__main__':
    cl_args = ParseArgs()
    num_classes = 64
    Pta=[20]
    epochs = 15
    batch_size = 512
    N_BS = 64
    N_MS = 64
    IsTrain=cl_args.IsTrain
    for Pt in Pta:   
        #### LOAD DATA
        h_train, h_test = read_data_sets(
                'MIMO_dataset/DeepMIMO_dataset_train'+str(Pt)+'.mat',
                'MIMO_dataset/DeepMIMO_dataset_test'+str(Pt)+'.mat',
                'MIMO_dataset/DeepMIMO_dataset_train_labels.mat',
                'MIMO_dataset/DeepMIMO_dataset_test_labels.mat',
                Flag_HDF5file)
        print('Using DeepMIMO_dataset_train'+str(Pt) +' dataset')

        n_samples, data_dim, _ = h_train.data.shape
        n_samples_test, _, _ = h_test.data.shape


        M_BSa = [2, 4, 8]
    
        for M_BS in M_BSa:
            M_MS=M_BS
            print('M_MS=M_BS='+str(M_BS))
            precoder=ConvPrecoder(data_dim,[N_BS,N_MS,M_BS,M_MS],num_classes)
        
            if IsTrain:
                precoder.train(10*h_train.data, h_train.labels, 10*h_test.data, h_test.labels, epochs, batch_size)
                precoder.save_weights('Saved_model/saved_weights_multi_'+str(Pt)+'_M_BS_'+str(M_BS)+'.h5')
            else:
                precoder.load_weights('Saved_model/saved_weights_multi_'+str(Pt)+'_M_BS_'+str(M_BS)+'.h5')

                def cat_loss(y_true, y_pred):
                    loss = K.mean(K.binary_crossentropy(y_true[:, 0:num_classes - 1], y_pred[:, 0:num_classes - 1]), axis=-1) \
                           + K.mean(K.binary_crossentropy(y_true[:, num_classes:2 * num_classes - 1], y_pred[:, num_classes:2 * num_classes - 1]), axis=-1)

                    return loss

                precoder.compile(optimizer='adam', loss=cat_loss)
        
            #### TEST
            encoded_data = precoder.inference(10*h_test.data)
            print(encoded_data.shape)
            index_pred_tx = np.zeros([n_samples_test, num_classes])
            index_pred_rx = np.zeros([n_samples_test, num_classes])
            y_pred_test_tx=encoded_data[:,:num_classes]
            y_pred_test_rx=encoded_data[:,num_classes:]
            label1_test_tx = h_test.labels[:, :num_classes]
            label1_test_rx = h_test.labels[:, num_classes:]
            precision_test_tx = 0
            precision_test_rx = 0
            for i1 in range(n_samples_test):
                idx = y_pred_test_tx[i1, :].argsort()[-3:][::-1]
                label_pred_tx = np.zeros([1, num_classes])
                for j1 in range(3):
                    label_pred_tx[0, idx[j1]] = 1
                label_ture = h_test.labels[i1, :num_classes]
                precision_test_tx = precision_test_tx + np.sum(np.logical_and(label_ture, label_pred_tx), axis=-1) / 3
                index_pred_tx[i1, :] = label_pred_tx
        
                idx = y_pred_test_rx[i1, :].argsort()[-3:][::-1]
                label_pred_rx = np.zeros([1, num_classes])
                for j1 in range(3):
                    label_pred_rx[0, idx[j1]] = 1
                label_ture = h_test.labels[i1, num_classes:]
                precision_test_rx = precision_test_rx + np.sum(np.logical_and(label_ture, label_pred_rx), axis=-1) / 3
                index_pred_rx[i1, :] = label_pred_rx
        
            precision_test_tx = precision_test_tx / n_samples_test
            precision_test_rx = precision_test_rx / n_samples_test
            print("Test Tx accuracy:" + str(precision_test_tx) + ", Rx accuracy:" + str(precision_test_rx))
           
            filename_db_save = 'Estimated_results/predicted_labels_TxRx_N64_test_Pt'+str(Pt)+'_m'+ str(M_BS) +'.mat'
            scio.savemat(filename_db_save,
                                 {'Tx_acc':precision_test_tx,'Rx_acc':precision_test_rx,'index_opt_tx': label1_test_tx, 'index_opt_rx': label1_test_rx, 'index_pred_tx': index_pred_tx, 'index_pred_rx':index_pred_rx})
