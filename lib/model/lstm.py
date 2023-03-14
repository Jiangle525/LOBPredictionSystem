from lib.model.base_model import Base_Model
from keras.models import Model, Input
from keras.layers import LSTM, Dense
from keras.optimizers import Adam


class LSTM_LOB(Base_Model):
    '''
        自定义LSTM模型类，
        实现了创建LSTM网络模型的方法
    '''

    def __init__(self, timesteps=100, features=40, number_of_lstm=64):
        super().__init__()
        self.model_name = "LSTM"
        self.create_model(timesteps, features, number_of_lstm)

    def create_model(self, T, NF, number_of_lstm):
        # 输入层
        input_lmd = Input(shape=(T, NF))

        # LSTM层
        conv_lstm = LSTM(number_of_lstm)(input_lmd)

        # 输出层
        out = Dense(3, activation='softmax')(conv_lstm)

        # 创建模型
        self.model = Model(inputs=input_lmd, outputs=out)

        # 自定义优化器
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)

        # 编译模型
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model
