import serial
import time
import torch
from djitellopy import Tello
import numpy as np
def command_ana(prelist):
    print(prelist)
    dict = {'Robot':0, 'Action':0}
    if 0 in prelist[0] or 4 in prelist[0]:
        dict['Robot'] = 1
    elif 1 in prelist[0] or 7 in prelist[0]:
        dict['Robot'] = 2
    elif 2 in prelist[0] or 6 in prelist[0]:
       dict['Robot'] = 3

    if 0 in prelist[1] or 4 in prelist[1]:
        dict['Action'] = 1
    elif 1 in prelist[1] or 7 in prelist[1]:  #
        dict['Action'] = 2
    elif 2 in prelist[1] or 6 in prelist[1]:
        dict['Action'] = 3
    elif 3 in prelist[1] or 5 in prelist[1]:  #
        dict['Action'] = 4
    return dict  # range from 1 to 4

def timecount(t):
    s = time.time()
    while time.time() - s <t:
        pass
        # print(time.time()-s)

class SerialSendMsg():
    def __init__(self, port, baud):
        try:
            self.ser = serial.Serial(port, baud)
            print('open success', port)
        except:
            print('open failed')

    def hc05_1(self, order):
        if self.ser.isOpen():
            try:
                self.ser.write((str(order[0]) + ' ' + order[1] + '\r\n').encode())  # windows下的串口需要+'\r\n'才能使数据脱离缓存区输出
                print('send success')
            except:
                print('send failed')
                pass

    def CAR(self, order):
        command, user_if = order[0], order[1]
        car_com = {1: "ONA", 2: "ONB", 3: "ONC", 4: "OND"}
        end_com = "ONF"
        print(type(command))
        if self.ser.isOpen():
            if command == 1 or command == 2:
                t = 1.5
                self.ser.write(car_com[command].encode())
                timecount(t)
                print('{} seconds run stop'.format(t))
                self.ser.write(end_com.encode())
            elif command == 3 or command == 4:
                t = 0.26
                self.ser.write(car_com[command].encode())
                timecount(t)
                print('{} seconds run stop'.format(t))
                self.ser.write(end_com.encode())
        else:
            print("Car serial open failed!")


    def DRONE(self, order):
        command, user_if = order[0], order[1]
        tello = Tello()
        tello.connect()
        response = tello.get_battery()
        print("Air drone's battery : ", response)
        time.sleep(2)
        tello.takeoff()
        time.sleep(2)
        if command == 1:
            tello.move_forward(int(100))
            time.sleep(3)
        elif command == 2:
            tello.move_left(int(100))
            time.sleep(3)
        elif command == 3:
            tello.move_right(int(100))
            time.sleep(3)
        elif command ==  4:
            tello.move_back(int(100))
            time.sleep(3)
        tello.land()

    def DRONE_SERIAL(self, order):
        ord, inf = order[0], order[1].numpy()
        inf = np.append(inf.reshape(-1), np.float64(ord))
        print(inf.shape)
        info = inf.astype(np.float64).tobytes()
        if self.ser.isOpen():
            try:
                self.ser.write(info)
                print('send success', 'send time ', time.time())
            except:
                print('send failed')
                pass

    def ARM(self, order):
        if self.ser.isOpen():
            try:
                self.ser.write(order)
                print('send success')
            except:
                print('send failed')
                pass

def Serial_Command(User_ifo, command_dict, ROBOT_COM={'ARM':'COM7', 'CAR':'COM18', 'AGENT':'COM12'}, baud_list = [9600, 9600, 57600]):  #{ }; {'Robot': 3, 'Action': 2}; {'ARM':'COM7', 'DOG':'COM8', 'AGENT':'COM9'}
    if command_dict['Robot'] == 1:
        print('{} is controling ARM'.format(User_ifo[0]))
        port2 = ROBOT_COM['ARM']
        baud = baud_list[0]
        order = [0x55, 0x55, 0x05, 0x06, command_dict['Action'], 0x01, 0x00]
        x = SerialSendMsg(port2, baud)
        x.ARM(order)

    elif command_dict['Robot'] == 2:
        print('{} is controling CAR'.format(User_ifo[0]))
        port1 = ROBOT_COM['CAR']
        baud = baud_list[1]
        order = [command_dict['Action'], User_ifo]
        x = SerialSendMsg(port1, baud)
        x.CAR(order)

    elif command_dict['Robot'] == 3:
        print('{} is controling FLY'.format(User_ifo[0]))
        User_ifo = torch.randn((1, 128))
        port1 = ROBOT_COM['AGENT']
        baud = baud_list[2]
        order = [command_dict['Action'], User_ifo]
        x = SerialSendMsg(port1, baud)
        x.DRONE_SERIAL(order)

if __name__ == '__main__':
    port1 = 'COM10'
    baud = 57600
    order = [3, 'state']
    port2 = 'COM7'
    baud = 9600
    order = [0x55, 0x55, 0x05, 0x06, 2, 0x01, 0x00]

    Serial_Command(['user'],{'Robot': 1, 'Action': 1})
