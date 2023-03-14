from PySide2.QtCore import Signal, QObject


class MySignal(QObject):
    progress_bar = Signal(int)  # 进度条信号
    tips = Signal(str)  # 提示区信号
    confusion_matrix = Signal(str)  # 混淆矩阵信号
    classification_report = Signal(str)  # 分类报告信号
    draw = Signal(int)  # 绘图信号


signal = MySignal()
