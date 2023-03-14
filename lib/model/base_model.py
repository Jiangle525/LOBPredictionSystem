import numpy as np
from keras.utils import np_utils
from keras.models import load_model
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Base_Model:
    ''' 封装了keras的Model对象，是本系统的基类模型 '''

    def __init__(self):
        self.model = None
        self.history = None
        self.result = []
        self.model_name = ''

    # 创建模型对象
    def creat_model(self):
        pass

    # 保存模型
    def save_model(self, file_name):
        try:
            self.model.save(file_name)
            return True
        except Exception:
            return False

    # 训练数据
    def train(self, trainX_CNN, trainY_CNN, epochs=10, batch_size=64, callbacks=None, verbose=0, validation_data=None):
        self.history = self.model.fit(trainX_CNN, trainY_CNN, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                                      verbose=verbose, validation_data=validation_data)

    # 预测数据
    def predict(self, x, batch_size=None, verbose=0, callbacks=None):
        self.result = self.model.predict(x, batch_size=batch_size, verbose=verbose, callbacks=callbacks)

    # 加载模型
    def load_model(self, file_name):
        self.model = load_model(file_name)

    # 加载数据
    def load_data(self, file_names_lst, label_position=4):
        ''' 返回结果：(data,label)
            数据有149行，n列，n>=1且为整数
            每一列为一个样本，倒数5行为标签，分别是k=10,20,30,50,100
            标签1表示百分比改变大于0.2%；
            标签2表示百分比改变介于-0.199% ~ 0.199%；
            标签3表示百分比改变小于或等于-0.02%
        '''
        try:
            # 取出第一个文件中的数据
            original_data = np.loadtxt(file_names_lst[0])
            # 依次加载后续数据并合并
            for file_name in file_names_lst[1:]:
                # 按行合并
                original_data = np.hstack((original_data, np.loadtxt(file_name)))

            return self.deal_data(original_data, label_position=label_position)

        except Exception:

            ''' #####处理文件读取异常##### '''

            exit()

    # 处理原始数据
    def deal_data(self, original_data, label_position, timesteps=100, features=40, label_begin=-5):
        '''返回结果：(data,label)
            data形状：(samples, timesteps, features,1)
            label形状：(samples,one-hot编码长度)
        '''
        # 转置矩阵
        data_tmp = original_data[:features, :].T
        label_tmp = original_data[label_begin:, :].T

        # 获取样本数
        samples = data_tmp.shape[0]
        # 生成零矩阵，shape为(samples - timesteps + 1, timesteps, features)
        data = np.zeros((samples - timesteps + 1, timesteps, features))
        for i in range(timesteps, samples + 1):
            data[i - timesteps] = data_tmp[i - timesteps:i, :]

        # 由于标签是1，2，3，采用one-hot编码时只需要3位即可，因此对标签统一的减一
        label = np.array(label_tmp)[timesteps - 1:samples] - 1
        label = np_utils.to_categorical(label[:, label_position])

        return data.reshape(data.shape + (1,)), label
