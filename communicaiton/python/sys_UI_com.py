import sys
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QPlainTextEdit, QProgressBar
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtGui import QFont, QPixmap
from communication.BL_EEGsingalCollect import com_connect
from communication.python.sub_data import main as epoc_col
from communication.serial_utility import *
from Model.REAL_TEST_main import Real_Test


class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):
        for i in range(1, 11):
            time.sleep(1)
            self.progress.emit(i * 10)
        self.finished.emit()


'''--------lite  COM 5---------'''
class MyThread1(threading.Thread):
    def __init__(self, label):
        threading.Thread.__init__(self)
        self.label = label

    def run(self):
        self.label.setText('User1 Robot Choosing Start')
        d1 = com_connect('COM5', '1_ind')
        self.label.setText('USER1 Robot Choosing Finished')

class MyThread2(threading.Thread):
    def __init__(self, label):
        threading.Thread.__init__(self)
        self.label = label

    def run(self):
        self.label.setText('User1 Action Choosing Start')
        print('LITE TIEM START',time.time())
        d2 = com_connect('COM5', '2_ind')

        self.label.setText('USER1 Action Choosing Finished')

'-------------------emotiv  NO COM------------'
class MyThread3(threading.Thread):
    def __init__(self, label):
        threading.Thread.__init__(self)
        self.label = label

    def run(self):
        self.label.setText('User2 Robot Choosing Start')
        d3 = epoc_col('1_ind')
        self.label.setText('USER2 Robot Choosing Finished')

class MyThread4(threading.Thread):
    def __init__(self, label):
        threading.Thread.__init__(self)
        self.label = label

    def run(self):
        self.label.setText('User2 Action Choosing Start')
        print('EPOC TIEM START',time.time())
        d4 = epoc_col('2_ind')
        self.label.setText('USER2 Action Choosing Finished')

'-------------------pink pro  COM19------------'
class MyThread5(threading.Thread):
    def __init__(self, label):
        threading.Thread.__init__(self)
        self.label = label

    def run(self):
        self.label.setText('User3 Robot Choosing Start')
        d5 = com_connect('COM19', '1_ind')
        self.label.setText('USER3 Robot Choosing Finished')

class MyThread6(threading.Thread):
    def __init__(self, label):
        threading.Thread.__init__(self)
        self.label = label

    def run(self):
        self.label.setText('User3 Action Choosing Start')
        print('PRO TIEM START',time.time())
        # 执行任务2的代码
        d6 = com_connect('COM19', '2_ind')

        self.label.setText('USER3 Action Choosing Finished')

'--------------SEND COMMAND--------------------'

class MyThread7(threading.Thread):
    def __init__(self, label):
        threading.Thread.__init__(self)
        self.label = label

    def run(self):
        self.label.setText('User1 Sending')
        file_path = r'C:\Users\欧阳\PycharmProjects\multi-brain-system\communication\data_record\brainlink_COM5'
        prelist = Real_Test(file_path)
        command_dict = command_ana(prelist)
        user_ifo = ['User1']
        x = Serial_Command(user_ifo, command_dict)
        self.label.setText('Command: {}'.format(command_dict))

class MyThread8(threading.Thread):
    def __init__(self, label):
        threading.Thread.__init__(self)
        self.label = label

    def run(self):
        self.label.setText('User2 Sending')
        file_path = r'C:\Users\欧阳\PycharmProjects\multi-brain-system\communication\data_record\Emotiv'
        prelist = Real_Test(file_path)
        command_dict = command_ana(prelist)
        user_ifo = ['User2']
        x = Serial_Command(user_ifo, command_dict)
        print(command_dict)
        self.label.setText('Command {}'.format(command_dict))

class MyThread9(threading.Thread):
    def __init__(self, label):
        threading.Thread.__init__(self)
        self.label = label

    def run(self):
        self.label.setText('User3 Sending')
        file_path = r'C:\Users\欧阳\PycharmProjects\multi-brain-system\communication\data_record\brainlink_COM19'
        prelist = Real_Test(file_path)
        command_dict = command_ana(prelist)
        user_ifo = ['User3']
        x = Serial_Command(user_ifo, command_dict)
        self.label.setText('Command: {}'.format(command_dict))



class MyWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.label = QLabel(self)
        self.label.setFixedSize(800, 340)

        pixmap = QPixmap('../command.jpg')
        scaled_pixmap = pixmap.scaled(self.label.size())

        self.label.setPixmap(scaled_pixmap)

        self.progress_bar1 = QLabel('User1')
        self.progress_bar2 = QLabel('User2')
        self.progress_bar3 = QLabel('User3')

        self.label1 = QLabel('')
        self.label2 = QLabel('')
        self.label3 = QLabel('')

        self.button1 = QPushButton('Robot choose')
        self.button1.setStyleSheet("font-size: 24px; color: white; background-color: lightblue; border-radius: 25px;")
        self.button1.setFixedSize(200, 50)
        self.button1.clicked.connect(self.on_button1_clicked)

        self.button2 = QPushButton('Action choose')
        self.button2.setStyleSheet("font-size: 24px; color: white; background-color: blue; border-radius: 25px;")
        self.button2.setFixedSize(200, 50)
        self.button2.clicked.connect(self.on_button2_clicked)

        self.button3 = QPushButton('Robot choose')
        self.button3.setStyleSheet("font-size: 24px; color: white; background-color: pink; border-radius: 25px;")
        self.button3.setFixedSize(200, 50)
        self.button3.clicked.connect(self.on_button3_clicked)

        self.button4 = QPushButton('Action choose')
        self.button4.setStyleSheet("font-size: 24px; color: white; background-color: red; border-radius: 20px;")
        self.button4.setFixedSize(200, 50)
        self.button4.clicked.connect(self.on_button4_clicked)

        self.button5 = QPushButton('Robot choose')
        self.button5.setStyleSheet("font-size: 24px; color: white; background-color: lightgreen; border-radius: 20px;")
        self.button5.setFixedSize(200, 50)
        self.button5.clicked.connect(self.on_button5_clicked)

        self.button6 = QPushButton('Action choose')
        self.button6.setStyleSheet("font-size: 24px; color: white; background-color: green; border-radius: 20px;")
        self.button6.setFixedSize(200, 50)
        self.button6.clicked.connect(self.on_button6_clicked)

        self.button7 = QPushButton('Command send')
        self.button7.setStyleSheet("font-size: 24px; color: white; background-color: darkblue; border-radius: 20px;")
        self.button7.setFixedSize(200, 50)
        self.button7.clicked.connect(self.on_button7_clicked)

        self.button8 = QPushButton('Command send')
        self.button8.setStyleSheet("font-size: 24px; color: white; background-color: darkred; border-radius: 20px;")
        self.button8.setFixedSize(200, 50)
        self.button8.clicked.connect(self.on_button8_clicked)

        self.button9 = QPushButton('Command send')
        self.button9.setStyleSheet("font-size: 24px; color: white; background-color: darkgreen; border-radius: 20px;")
        self.button9.setFixedSize(200, 50)
        self.button9.clicked.connect(self.on_button9_clicked)

        vbox00 = QVBoxLayout()
        vbox00.addWidget(self.progress_bar1)
        vbox00.addWidget(self.progress_bar2)
        vbox00.addWidget(self.progress_bar3)

        vbox0 = QVBoxLayout()
        vbox0.addWidget(self.label1)
        vbox0.addWidget(self.label2)
        vbox0.addWidget(self.label3)

        vbox1 = QVBoxLayout()
        vbox1.addWidget(self.button1)
        vbox1.addWidget(self.button3)
        vbox1.addWidget(self.button5)


        vbox2 = QVBoxLayout()
        vbox2.addWidget(self.button2)
        vbox2.addWidget(self.button4)
        vbox2.addWidget(self.button6)

        vbox3 = QVBoxLayout()
        vbox3.addWidget(self.button7)
        vbox3.addWidget(self.button8)
        vbox3.addWidget(self.button9)


        hbox = QHBoxLayout()
        hbox.addLayout(vbox00)
        hbox.addLayout(vbox0)
        hbox.addLayout(vbox1)
        hbox.addLayout(vbox2)
        hbox.addLayout(vbox3)

        vbox = QVBoxLayout()
        vbox.addWidget(self.label)
        vbox.addLayout(hbox)

        self.setLayout(vbox)


    def on_button1_clicked(self):
        thread1 = MyThread1(self.label1)
        thread1.start()

    def on_button2_clicked(self):
        thread2 = MyThread2(self.label1)
        thread2.start()

    def on_button3_clicked(self):
        thread3 = MyThread3(self.label2)
        thread3.start()

    def on_button4_clicked(self):
        thread4 = MyThread4(self.label2)
        thread4.start()

    def on_button5_clicked(self):
        thread5 = MyThread5(self.label3)
        thread5.start()

    def on_button6_clicked(self):
        thread6 = MyThread6(self.label3)
        thread6.start()

    def on_button7_clicked(self):
        thread7 = MyThread7(self.label1)
        thread7.start()

    def on_button8_clicked(self):
        thread8 = MyThread8(self.label2)
        thread8.start()

    def on_button9_clicked(self):
        thread9 = MyThread9(self.label3)
        thread9.start()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    font = QFont("Arial", 12)
    app.setFont(font)

    window = MyWindow()
    window.setStyleSheet("background-color: white;")
    window.resize(400, 400)

    window.show()
    sys.exit(app.exec_())
