import os
from keras.layers import Input, Conv2D, MaxPooling2D, Reshape, AveragePooling2D, Dense, Activation
import h5py
import numpy as np
import keras
import keras.models as model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras.backend as kb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KERAS_BACKEND'] = 'tensorflow'

'''
    文献中MCNet网络结构识别
    数据集：2018.01a DATASETS
    包含24中调制方式，每种调制方式有[-20:2:30]共25个SNRs，每个snr有4096条数据，每条数据I/Q两路数据，每路数据有1024个点
    数据集的大小为  2555904*2*1024


    MCNet网络结构：
            out-dim
    Input: 2*1024*1
    conv:  2*512*64
    pool:  2*256*64
    preB:  2*128*64
    pool:  2*64*64
    M_BP:  2*32*128
    add:   2*32*128
    M_Bk:  2*32*128
    add:   2*32*128
    M_BP:  2*16*128
    add:   2*16*128
    M_Bk:  2*16*128
    add:   2*16*128
    M_BP:  2*8*128
    add:   2*8*128
    M_Bk:  2*8*256
    concat:
    pool:

    dense:
    softmax:

    classification                     

'''

# 分割数据集

for itr in range(9):
    filename = '/data/yqwan/DATASET/20210119/new_snr_' + str(itr) + '.h5'
    print(filename)
    f = h5py.File(filename, 'r')
    x_data = f['X'][:]
    y_data = f['Y'][:]
    z_data = f['Z'][:]
    print(x_data.shape)
    print(y_data.shape)
    print(z_data.shape)
    f.close()

    n_examples = x_data.shape[0]
    train_index = np.random.choice(range(0, n_examples), size=int(0.7 * n_examples), replace=False)
    test_index = list(set(range(0, n_examples)) - set(train_index))

    if itr == 0:
        x_train = x_data[train_index]
        y_train = y_data[train_index]
        z_train = z_data[train_index]
        x_test = x_data[test_index]
        y_test = y_data[test_index]
        z_test = z_data[test_index]
    else:
        x_train = np.vstack((x_train, x_data[train_index]))
        y_train = np.vstack((y_train, y_data[train_index]))
        z_train = np.vstack((z_train, z_data[train_index]))
        x_test = np.vstack((x_test, x_data[test_index]))
        y_test = np.vstack((y_test, y_data[test_index]))
        z_test = np.vstack((z_test, z_data[test_index]))

print('训练集X维度：', x_train.shape)
print('训练集Y维度：', y_train.shape)
print('训练集Z维度：', z_train.shape)
print('测试集X维度：', x_test.shape)
print('测试集Y维度：', y_test.shape)
print('测试集Z维度：', z_test.shape)


# pre_block模块
def pre_block(xm, mcSeq, pool_size):
    print('进入pre-block模块')
    base = xm
    xm0 = Conv2D(32, (3, 1), padding='same', name=mcSeq + "_pre_block_model_conv00", kernel_initializer='glorot_normal', data_format='channels_first')(base)
    xm0 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format='channels_first')(xm0)
    xm1 = Conv2D(32, (3, 1), padding='same', name=mcSeq + "_pre_block_model_conv01", kernel_initializer='glorot_normal', data_format='channels_first')(base)
    xm = kb.concatenate([xm0, xm1], axis=2)
    # concat(xm0, xm1)
    return xm


# m_block模块
def m_block(xm, mcSeq):
    print('进入m-block模块')
    base = xm
    base_xm = Conv2D(32, (1, 1), padding='same', name=mcSeq + "_m_block_conv1", kernel_initializer='glorot_normal', data_format='channels_first')(base)
    xm0 = Conv2D(32, (3, 1), padding='same', name=mcSeq + "_m_block_pool", kernel_initializer='glorot_normal', data_format='channels_first')(base_xm)
    xm1 = Conv2D(32, (3, 1), padding='same', name=mcSeq + "_m_block_model_conv01", kernel_initializer='glorot_normal', data_format='channels_first')(base_xm)
    xm2 = Conv2D(32, (1, 1), padding='same', name=mcSeq + "_m_block_model_conv02", kernel_initializer='glorot_normal', data_format='channels_first')(base_xm)
    xm = kb.concatenate([xm0, xm1], axis=2)
    xm = kb.concatenate([xm, xm2], axis=2)
    # concat(xm0, xm1, xm2)
    return xm


# m_block_pool模块
def m_block_p(xm, mcSeq, pool_size):
    print('进入m_block_pool模块')
    base = xm
    base_xm = Conv2D(32, (1, 1), padding='same', name=mcSeq + "_m_block_p_conv1", kernel_initializer='glorot_normal', data_format='channels_first')(base)
    xm0 = Conv2D(32, (3, 1), padding='same', name=mcSeq + "_pre_block_p_model_conv00", kernel_initializer='glorot_normal', data_format='channels_first')(base_xm)
    xm0 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format='channels_first')(xm0)
    xm1 = Conv2D(32, (3, 1), padding='same', name=mcSeq + "_pre_block_p_model_conv01", kernel_initializer='glorot_normal', data_format='channels_first')(base_xm)
    xm2 = Conv2D(32, (1, 1), padding='same', name=mcSeq + "_pre_block_p_model_conv02", kernel_initializer='glorot_normal', data_format='channels_first')(base_xm)
    xm = kb.concatenate([xm0, xm1], axis=2)
    xm = kb.concatenate([xm, xm2], axis=2)
    # concat(xm0, xm1, xm2)
    return xm


# 建模

classes = ['AM',
           'FM',
           'ASK',
           'BPSK',
           'QPSK',
           '8PSK',
           '16QAM',
           'GMSK',
           'APSK']

in_shp = list(x_train.shape[1:])
print(in_shp)

# input
xm_input = Input(in_shp)
xm = Reshape([1, 1024, 2], input_shape=in_shp)(xm_input)

# conv
xm = Conv2D(64, (3, 1), padding='same', name='conv0', kernel_initializer='glorot_normal', data_format='channels_first')(xm)

# pool
xm = MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='valid', data_format='channels_first')(xm)


# keras.backend.concatenate

# pre-block
mcSeq = 'mc_net01'  # 可以自行定义，每一层网络的名字。。。
pool_size = (3, 1)
xm = pre_block(xm, mcSeq, pool_size)
xm_tmp1 = MaxPooling2D(pool_size=(3, 1), strides=(3, 1), padding='valid', data_format='channels_first')(xm)

# pool
xm = MaxPooling2D(pool_size=(3, 1), strides=(3, 1), padding='valid', data_format='channels_first')(xm)

# M-block-p
mcSeq = 'mc_net02'
xm = m_block_p(xm, mcSeq, pool_size)

# add
xm = kb.concatenate([xm, xm_tmp1], axis=2)
xm_tmp2 = xm

# M-block
mcSeq = 'mc_net03'
xm = m_block(xm, mcSeq)

# add
xm = kb.concatenate([xm, xm_tmp2], axis=2)
xm_tmp3 = xm
xm_tmp3_pool = MaxPooling2D(pool_size=(3, 1), strides=(3, 1), padding='valid', data_format='channels_first')(xm_tmp3)

# M-block-p
mcSeq = 'mc_net04'
xm = m_block_p(xm, mcSeq, pool_size)

# add
xm = kb.concatenate([xm, xm_tmp3_pool], axis=2)
xm_tmp4 = xm
# M-block
mcSeq = 'mc_net05'
xm = m_block(xm, mcSeq)

# add
xm = kb.concatenate([xm, xm_tmp4], axis=2)
xm_tmp5 = xm
xm_tmp5_pool = MaxPooling2D(pool_size=(3, 1), strides=(3, 1), padding='valid', data_format='channels_first')(xm_tmp5)

# M-block-p
mcSeq = 'mc_net06'
xm = m_block_p(xm, mcSeq, pool_size)
# add
xm = kb.concatenate([xm, xm_tmp5_pool], axis=2)
xm_tmp6 = xm

# M-block
mcSeq = 'mc_net07'
xm = m_block(xm, mcSeq)
xm_tmp7 = xm

# concat
xm = kb.concatenate([xm, xm_tmp7], axis=2)

# avg-pool
xm = AveragePooling2D(pool_size=(3, 1), strides=(3, 1), padding='valid', data_format='channels_first')( xm)

# dense fc
xm = Dense(len(classes), kernel_initializer='glorot_normal', name="dense3")(xm)

# softmax
xm = Activation('softmax')(xm)

model = model.Model(inputs=xm_input, outputs=xm)
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
model.summary()

# 训练模型
print("训练开始")
filepath = '/data/yqwan/ModelData/mcnet001.h5'
history = model.fit(x_train, y_train,
                    batch_size=1024,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    callbacks=[
                        keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto'),
                        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
                    ])
print("训练结束了")

# model.load_weights(filepath)

# 绘图测试收敛性
plt.figure(1)
# plt.subplot(2, 1, 1)
plt.title('CNN_LOSS')
val_loss_list = history.history['val_loss']
loss_list = history.history['loss']
plt.plot(range(len(loss_list)), loss_list, label="loss")
plt.plot(range(len(val_loss_list)), val_loss_list, label="val_loss")
plt.grid(True)
plt.axis('tight')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()

# plt.subplot(2, 1, 2)
acc_list = history.history['accuracy']
val_acc_list = history.history['val_accuracy']
plt.title('CNN_ACC')
plt.plot(range(len(acc_list)), acc_list, label="acc")
plt.plot(range(len(val_acc_list)), val_acc_list, label="val_acc")
plt.grid(True)
plt.axis('tight')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
# plt.savefig()
plt.show()

print("game over")




