from lib.model.base_model import Base_Model
from keras.models import Model, Input
from keras.layers import Dense, Flatten, Conv2D, LeakyReLU
from keras.optimizers import Adam


class CNN_LOB(Base_Model):
    '''
        自定义CNN模型类，
        实现了创建CNN网络模型的方法
    '''
    def __init__(self, timesteps=100, features=40, number_of_lstm=64):
        super().__init__()
        self.model_name = "CNN"
        self.create_model(timesteps, features, number_of_lstm)

    def create_model(self, T, NF, number_of_cnn):
        # 输入层
        input_lmd = Input(shape=(T, NF, 1))

        # 卷积块1
        conv_first1 = Conv2D(number_of_cnn, (1, 2), strides=(1, 2))(input_lmd)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(number_of_cnn, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(number_of_cnn, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        # 卷积块2
        conv_first1 = Conv2D(number_of_cnn, (1, 2), strides=(1, 2))(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(number_of_cnn, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(number_of_cnn, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        # 卷积块3
        conv_first1 = Conv2D(number_of_cnn, (1, 10))(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(number_of_cnn, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)
        conv_first1 = Conv2D(number_of_cnn, (4, 1), padding='same')(conv_first1)
        conv_first1 = LeakyReLU(alpha=0.01)(conv_first1)

        # 使用Flatten展平
        conv_first1 = Flatten()(conv_first1)

        # 连接到输出层
        out = Dense(3, activation='softmax')(conv_first1)

        # 创建模型
        self.model = Model(inputs=input_lmd, outputs=out)

        # 自定义优化器
        adam = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1)

        # 编译模型
        self.model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model
