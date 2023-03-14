from lib.model.base_model import Base_Model
from keras.layers import Input, LeakyReLU, MaxPooling2D, concatenate, Reshape, LSTM, Conv2D, Dense
from keras.models import Model
from keras.optimizers import Adam


class CNN_LSTM_LOB(Base_Model):
    '''
        自定义CNN_LSTM模型类，
        实现了创建CNN_LSTM网络模型的方法
    '''
    def __init__(self, timesteps=100, features=40, number_of_lstm=64):
        super().__init__()
        self.model_name = "CNN_LSTM"
        self.create_model(timesteps, features, number_of_lstm)

    def create_model(self, timesteps, features, number_of_lstm):
        # 输入层
        input_ = Input(shape=(timesteps, features, 1))

        # 卷积块1
        conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(input_)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        # 卷积块2
        conv_first1 = Conv2D(32, (1, 2), strides=(1, 2))(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        # 卷积块3
        conv_first1 = Conv2D(32, (1, 10))(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(32, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        # Inception模块
        convsecond_1 = Conv2D(64, (1, 1), padding='same')(conv_first1)
        convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)
        convsecond_1 = Conv2D(64, (3, 1), padding='same')(convsecond_1)
        convsecond_1 = LeakyReLU(alpha=0.01)(convsecond_1)

        convsecond_2 = Conv2D(64, (1, 1), padding='same')(conv_first1)
        convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)
        convsecond_2 = Conv2D(64, (5, 1), padding='same')(convsecond_2)
        convsecond_2 = LeakyReLU(alpha=0.01)(convsecond_2)

        convsecond_3 = MaxPooling2D((3, 1), strides=(1, 1), padding='same')(conv_first1)
        convsecond_3 = Conv2D(64, (1, 1), padding='same')(convsecond_3)
        convsecond_3 = LeakyReLU(alpha=0.01)(convsecond_3)

        # 组合Inception模块
        convsecond_output = concatenate([convsecond_1, convsecond_2, convsecond_3], axis=3)

        # 使用dropout
        conv_reshape = Reshape((int(convsecond_output.shape[1]), int(convsecond_output.shape[3])))(convsecond_output)

        # LSTM层
        conv_lstm = LSTM(number_of_lstm)(conv_reshape)

        # 输出层
        out = Dense(3, activation='softmax')(conv_lstm)

        # 创建模型
        self.model = Model(inputs=input_, outputs=out)

        # 自定义优化器
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)

        # 编译模型
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model
