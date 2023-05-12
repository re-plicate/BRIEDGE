
import matplotlib.pyplot as plt
import time
import xlrd
import math
import re
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import serial



def cleartxt(filename):
    try:
        with open(filename, 'w') as nao:
            nao.truncate()
        print('initial txt normally')
    except:
        print('can not creat a txt')
        pass


def com_connect(com, instru_index):
    print('collecting data')
    a = time.time()
    filename = r'****\multi-brain-system\communication\data_record\brainlink_{}/{}_brainlink'.format(com, instru_index)
    cleartxt(filename)
    t = serial.Serial(com, 57600)

    b = t.read(3)
    vaul = []
    largedata=[]
    i = 0
    y = 0
    p = 0
    ii = 0
    while b[0] != 170 or b[1] != 170 or b[2] != 4:
        b = t.read(3)
        print(b)
    if b[0] == b[1] == 170 and b[2] == 4:
        a = b + t.read(5)
        print(a)

        start_time = time.time()
        if a[0] == 170 and a[1] == 170 and a[2] == 4 and a[3] == 128 and a[4] == 2:  #
            while 1:

                i = i + 1
                a = t.read(8)
                sum = ((0x80 + 0x02 + a[5] + a[6]) ^ 0xffffffff) & 0xff
                if a[0] == a[1] == 170 and a[2] == 32:
                    y = 1  #
                else:
                    y = 0
                if a[0] == 170 and a[1] == 170 and a[2] == 4 and a[3] == 128 and a[4] == 2:
                    p = 1
                else:
                    p = 0
                if sum != a[7] and y != 1 and p != 1:
                    print("wrroy1")
                    b = t.read(3)
                    c = b[0]
                    d = b[1]
                    e = b[2]
                    print(b)
                    while c != 170 or d != 170 or e != 4:
                        c = d
                        d = e
                        e = t.read()
                        print("c:")
                        print(c)
                        print("d:")
                        print(d)
                        print("e:")
                        print(e)
                        if c == (b'\xaa' or 170) and d == (b'\xaa' or 170) and e == b'\x04':
                            g = t.read(5)
                            print(g)
                            if c == b'\xaa' and d == b'\xaa' and e == b'\x04' and g[0] == 128 and g[1] == 2:
                                a = t.read(8)
                                print(a)
                                break


                if a[0] == 170 and a[1] == 170 and a[2] == 4 and a[3] == 128 and a[4] == 2:
                    high = a[5]
                    low = a[6]
                    #                print(a)
                    rawdata = (high << 8) | low
                    if rawdata > 32768:
                        rawdata = rawdata - 65536
                    sum = ((0x80 + 0x02 + high + low) ^ 0xffffffff) & 0xff
                    if sum == a[7]:
                        vaul.append(rawdata)
                    if sum != a[7]:
                        print("wrroy2")
                        b = t.read(3)
                        c = b[0]
                        d = b[1]
                        e = b[2]
                        #                    print(b)
                        while c != 170 or d != 170 or e != 4:
                            c = d
                            d = e
                            e = t.read()
                            if c == b'\xaa' and d == b'\xaa' and e == b'\x04':
                                g = t.read(5)
                                print(g)
                                if c == b'\xaa' and d == b'\xaa' and e == b'\x04' and g[0] == 128 and g[1] == 2:
                                    a = t.read(8)
                                    print(a)
                                    break
                if a[0] == a[1] == 170 and a[2] == 32:
                    c = a + t.read(28)
                    print(len(vaul))
                    if len(vaul)==512:
                        ii+=1
                        attitude= 'focus' if c[-4]>50 else 'relax'
                        print('collecting the num of {} data is {}'.format(ii,c[-4]),attitude)
                        for i in range(1):
                            largedata.append(c[-4])
                            print(largedata)
                            with open(filename, 'a') as file_object:
                                file_object.write(str(c[-4]))
                                file_object.write(" ")
                    if len(largedata)==15:
                        out_time = time.time() - start_time
                        print('processing time',out_time,'s')
                        print('finish')
                        break
                    vaul=[]
    return largedata

if __name__ == '__main__':
    com = 'COM6'
    col_d = com_connect(com, 'task_name')
    # print(col_d)