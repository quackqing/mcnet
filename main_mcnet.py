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
    测试使用两次1*1卷积的维数情况

'''


# 从总的数据集中获得训练集和测试集
for itr in range(1):

    filename = '/Users/yqwan/data/DataSets/ExtractDataset/part' + str(itr) + '.h5'
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

# 验证数据
# sample_idx = np.random.randint(74547)
# print(sample_idx)
# print('snr: ', z_train[sample_idx])
# print('标签y: ', y_train[sample_idx])
#
# plt_data = x_train[sample_idx].T
# plt.figure(figsize=(15, 5))
# plt.plot(plt_data[0], 'blue')
# plt.plot(plt_data[1], 'red')
# plt.show()


# 全局变量，合并维数
concat_axis = 3

# pre_block模块
def pre_block(xm, conv1_size, conv2_size, pool_size, mcSeq):
    print('进入pre-block模块')
    base = xm
    xm0 = Conv2D(32, conv1_size, padding='same', name=mcSeq + "_pre_block_conv1", kernel_initializer='glorot_normal', data_format='channels_last')(base)
    xm0 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format='channels_last')(xm0)
    xm1 = Conv2D(32, conv2_size, padding='same', name=mcSeq + "_pre_block_conv2", kernel_initializer='glorot_normal', data_format='channels_last')(base)
    xm1 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format='channels_last')(xm1)
    xm = kb.concatenate([xm0, xm1], axis=concat_axis)
    # concat(xm0, xm1)
    return xm


# m_block模块
def m_block(xm, filters_size01, filters_size02, filters_size03, conv0_size, conv1_size, conv2_size, conv3_size, mcSeq):
    print('进入m-block模块')
    base = xm
    base_xm = Conv2D(filters_size01, conv0_size, padding='same', name=mcSeq + "_m_block_conv0", kernel_initializer='glorot_normal', data_format='channels_last')(base)
    xm0 = Conv2D(filters_size02, conv1_size, padding='same', name=mcSeq + "_m_block_conv1", kernel_initializer='glorot_normal', data_format='channels_last')(base_xm)
    xm1 = Conv2D(filters_size02, conv2_size, padding='same', name=mcSeq + "_m_block_conv2", kernel_initializer='glorot_normal', data_format='channels_last')(base_xm)
    xm2 = Conv2D(filters_size03, conv3_size, padding='same', name=mcSeq + "_m_block_conv3", kernel_initializer='glorot_normal', data_format='channels_last')(base_xm)
    xm = kb.concatenate([xm0, xm1], axis=concat_axis)
    xm = kb.concatenate([xm, xm2], axis=concat_axis)
    # concat(xm0, xm1, xm2)
    return xm


# m_block_pool模块
def m_block_p(xm, conv0_size, conv1_size, conv2_size, conv3_size, pool_size, mcSeq):
    print('进入m_block_pool模块')
    base = xm
    base_xm = Conv2D(32, conv0_size, padding='same', name=mcSeq + "_m_block_p_conv0", kernel_initializer='glorot_normal', data_format='channels_last')(base)
    xm0 = Conv2D(48, conv1_size, padding='same', name=mcSeq + "_pre_block_p_conv1", kernel_initializer='glorot_normal', data_format='channels_last')(base_xm)
    xm0 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format='channels_last')(xm0)
    xm1 = Conv2D(48, conv2_size, padding='same', name=mcSeq + "_pre_block_p_conv2", kernel_initializer='glorot_normal', data_format='channels_last')(base_xm)
    xm1 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format='channels_last')(xm1)
    xm2 = Conv2D(32, conv3_size, padding='same', name=mcSeq + "_pre_block_p_conv3", kernel_initializer='glorot_normal', data_format='channels_last')(base_xm)
    xm2 = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding='valid', data_format='channels_last')(xm2)
    xm = kb.concatenate([xm0, xm1], axis=concat_axis)
    xm = kb.concatenate([xm, xm2], axis=concat_axis)
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

# input层Input()  1024*2
xm_input = Input(in_shp)

# Reshape() [1,1024,2]
xm = Reshape([2, 1024, 1], input_shape=in_shp)(xm_input)

# 额外加的pre_conv，目的是给他搞成[64, 1024, 2]
# data_format_type = 'channels_last'  # 其实默认的也是channels_last ,强调时则需要data_format='channels_first'
xm = Conv2D(64, kernel_size=(1, 1), padding='same', name='pre_conv', kernel_initializer='glorot_normal', data_format='channels_last')(xm)

# 额外加一个MaxPooling2D()
xm = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid', data_format='channels_last')(xm)

# conv name-->conv0
# 先进行conv0进行卷积，内核大小为[7，3]/输入维度为[64, 1024, 2]--
xm = Conv2D(64, kernel_size=(3, 7), padding='same', name='conv0', kernel_initializer='glorot_normal', data_format='channels_last')(xm)

# pool0----->MaxPooling2D() 最大池化层
xm = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid', data_format='channels_last')(xm)

# pre-block  pre-B
# pre_block(xm,conv1_size,conv2_size,pool_size,mcSeqName)
mcPreBName = 'mc_net01'  # 可以自行定义，每一层sub网络的名字。。。
pre_B_conv1_size = (1, 3)
pre_B_conv2_size = (3, 1)
pre_B_pool_size = (1, 2)
xm = pre_block(xm, pre_B_conv1_size, pre_B_conv2_size, pre_B_pool_size, mcPreBName)


# jumpPool1_size----->MaxPooling2D
# 这儿的结构还有些问题，需要在推敲推敲
# jumpPool1_size = (1, 2)
# jumpStrides1_size = (1, 1)
# xm_tmp1 = MaxPooling2D(pool_size=jumpPool1_size, strides=jumpPool1_size, padding='valid', data_format='channels_last')(xm)

# pool1----->MaxPooling2D()
xm = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding='valid', data_format='channels_last')(xm)

# m_Bp1----->m_block_p
# m_block_p(xm, conv0_size, conv1_size, conv2_size, conv3_size, pool_size, mcSeq)
mcMBp1Name = 'mc_net02'
m_Bp1_conv0_size = (1, 1)
m_Bp1_conv1_size = (3, 1)
m_Bp1_conv2_size = (1, 3)
m_Bp1_conv3_size = (1, 1)
m_Bp1_pool_size = (1, 2)
xm = m_block_p(xm, m_Bp1_conv0_size, m_Bp1_conv1_size, m_Bp1_conv2_size, m_Bp1_conv3_size, m_Bp1_pool_size, mcMBp1Name)

# add
# xm = kb.concatenate([xm, xm_tmp1], axis=concat_axis)
xm_tmp2 = xm

# m_B1----->m_block
# m_block(xm, conv0_size, conv1_size, conv2_size, conv3_size, mcSeq)
mcMB1Name = 'mc_net03'
m_B1_filter_size01 = 32
m_B1_filter_size02 = 48
m_B1_filter_size03 = 32
m_B1_conv0_size = (1, 1)
m_B1_conv1_size = (1, 3)
m_B1_conv2_size = (3, 1)
m_B1_conv3_size = (1, 1)
xm = m_block(xm, m_B1_filter_size01, m_B1_filter_size02, m_B1_filter_size03, m_B1_conv0_size, m_B1_conv1_size, m_B1_conv2_size, m_B1_conv3_size, mcMB1Name)

# add 表示各个元素相加，而非拼接
# xm = kb.concatenate([xm, xm_tmp2], axis=concat_axis)
xm = keras.layers.Add()([xm, xm_tmp2])   # xm = keras.layers.add([xm, xm_tmp2])
xm_tmp3 = xm
# poolJump1----->MaxPooling2D()
jumpPool2_size = (1, 2)
jumpStrides2_size = (1, 2)
xm_tmp3_pool = MaxPooling2D(pool_size=jumpPool2_size, strides=jumpStrides2_size, padding='valid', data_format='channels_last')(xm_tmp3)

# M-block-p
mcMBp2Name = 'mc_net04'
m_Bp2_conv0_size = (1, 1)
m_Bp2_conv1_size = (1, 3)
m_Bp2_conv2_size = (3, 1)
m_Bp2_conv3_size = (1, 1)
m_Bp2_pool_size = (1, 2)
xm = m_block_p(xm, m_Bp2_conv1_size, m_Bp2_conv1_size, m_Bp2_conv2_size, m_Bp2_conv3_size, m_Bp2_pool_size, mcMBp2Name)

# add
# xm = kb.concatenate([xm, xm_tmp3_pool], axis=concat_axis)
xm = keras.layers.Add()([xm, xm_tmp3_pool])
xm_tmp4 = xm
# M-block
mcMB2Name = 'mc_net05'
m_B2_filter_size01 = 32
m_B2_filter_size02 = 48
m_B2_filter_size03 = 32
m_B2_conv0_size = (1, 1)
m_B2_conv1_size = (1, 3)
m_B2_conv2_size = (3, 1)
m_B2_conv3_size = (1, 3)
xm = m_block(xm, m_B2_filter_size01, m_B2_filter_size02, m_B2_filter_size03, m_B2_conv0_size, m_B2_conv1_size, m_B2_conv2_size, m_B2_conv3_size, mcMB2Name)


# add
# xm = kb.concatenate([xm, xm_tmp4], axis=concat_axis)
xm = keras.layers.Add()([xm, xm_tmp4])
xm_tmp5 = xm
jumpPool3_size = (1, 2)
jumpStrides3_size = (1, 2)
xm_tmp5_pool = MaxPooling2D(pool_size=jumpPool3_size, strides=jumpStrides2_size, padding='valid', data_format='channels_last')(xm_tmp5)


# M-block-p
mcMBp3Name = 'mc_net06'
m_Bp3_conv0_size = (1, 1)
m_Bp3_conv1_size = (1, 3)
m_Bp3_conv2_size = (3, 1)
m_Bp3_conv3_size = (1, 1)
m_Bp3_pool_size = (1, 2)
xm = m_block_p(xm, m_Bp3_conv0_size, m_Bp3_conv1_size, m_Bp3_conv2_size, m_Bp3_conv3_size, m_Bp3_pool_size, mcMBp3Name)


# add
# xm = kb.concatenate([xm, xm_tmp5_pool], axis=concat_axis)
xm = keras.layers.Add()([xm, xm_tmp5_pool])
xm_tmp6 = xm


# M-block
mcMB3Name = 'mc_net07'
m_B3_filter_size01 = 32
m_B3_filter_size02 = 96
m_B3_filter_size03 = 64
m_B3_conv0_size = (1, 1)
m_B3_conv1_size = (1, 3)
m_B3_conv2_size = (3, 1)
m_B3_conv3_size = (1, 3)
xm = m_block(xm, m_B3_filter_size01, m_B3_filter_size02, m_B3_filter_size03, m_B3_conv0_size, m_B3_conv1_size, m_B3_conv2_size, m_B3_conv3_size, mcMB3Name)
xm_tmp7 = xm


# concat
xm = kb.concatenate([xm, xm_tmp6], axis=concat_axis)

# pool2----->avg-pool----->AveragePooling2D()
xm = AveragePooling2D(pool_size=(2, 8), strides=(1, 2), padding='valid', data_format='channels_last')(xm)

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
filepath = '/Users/yqwan/models/mcnet001.h5'
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






