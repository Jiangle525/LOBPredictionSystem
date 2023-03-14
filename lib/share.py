from lib.settings import option_setting
from lib.model.base_model import Base_Model


class SI:
    ''' 公共信息类 '''
    mainWin = None  # 主窗口对象
    model = Base_Model()  # 自定义模型对象
    option_setting = option_setting  # 训练模型的选项设置
    train_file_lst = None  # 训练数据集地址
    predict_file_lst = None  # 待预测数据集地址
    default_batch_size = 64  # 默认批量大小
    default_epochs = 10  # 默认训练轮数
    default_h_value = 1  # 默认k值
    x_train = None  # 训练数据
    y_train = None  # 训练数据的标签
    x_data = None  # 待预测数据
    y_true = None  # 待预测数据的标签
