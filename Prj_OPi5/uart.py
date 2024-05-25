
import serial
from enum import Enum

serial_name="/dev/ttyS6" #"/dev/ttyUSB0"
COMMAND_BEGIN=0xee  #命令帧头
CHECK_INIT_VALUE=0x21   #用于计算校验码

class Car_Cmd(Enum):
    RUN=0x01
    STOP=0x02
    TURN_LEFT_RUN=0x03   #前进中左转
    TURN_RIGHT_RUN=0x04
    TURN_LEFT=0x05    #原地
    TURN_RIGHT=0x06
    RUN_BACK=0x07
    RESET=0X11

class My_Serial:
    def __init__(self,port,baud):
        try:
            self.Serial_cmd = serial.Serial(port, baud)  # 初始化串口
            if not self.Serial_cmd.isOpen():
                self.Serial_cmd.open()                  # 确保端口打开
            print(f"Connected to {self.Serial_cmd.name}")

        except serial.SerialException as e:
            print(f"Serial port error: {e}")

    @staticmethod
    def get_and_append_checksum(command_list:list[int]):
        checksum=CHECK_INIT_VALUE
        for command in command_list:
            checksum=checksum ^ command
        command_list.append(checksum)

    def detect_and_auto_run_begin(self) -> None:
        """
        用于开启小车向前进，并发送信号，开启边缘检测
        """

        self.auto_status = True  # 说明此时处于小车自动运行状态

        # 初始化小车转弯状态
        self.left_status = INIT_TURN_LEFT_FLAG

        self.process_forward()  # 小车走直线运行
    
    def process_forward(self) -> None:
        """
        小车向前直线运行
        """
        self.process_command(Car_Cmd.RUN,0x00)

    def process_stop(self) -> None:
        """
        小车停止
        """
        self.process_command(Car_Cmd.STOP,0x00)

    def process_run_turn_left(self,speed) -> None:
        """
        小车前进中左转
        """
        self.process_command(Car_Cmd.TURN_LEFT_RUN,speed)

    def process_run_turn_right(self,speed) -> None:
        """
        小车前进中右转
        """
        self.process_command(Car_Cmd.TURN_RIGHT_RUN,speed)

    def process_situ_turn_left(self,speed) -> None:
        """
        小车原地左转
        """
        self.process_command(Car_Cmd.TURN_LEFT,speed)

    def process_situ_turn_right(self,speed) -> None:
        """
        小车原地右转
        """
        self.process_command(Car_Cmd.TURN_RIGHT,speed)

    def process_reset(self) -> None:
        """
        小车状态重置
        """
        self.process_command(Car_Cmd.COMMAND_RESET,0x00)

    #发送小车控制命令
    def process_command(self,command:Car_Cmd,speed):
        # 设置命令列表
        command_list=[COMMAND_BEGIN,COMMAND_BEGIN,command.value,speed]
        #计算校验码并添加进列表
        self.get_and_append_checksum(command_list)
        #发送命令
        self.Serial_cmd.write(bytes(command_list))

def process_serial(serial_name,queue):
    serial=My_Serial(serial_name,115200)
    while True:
        cmd=queue.get()
        speed=queue.get()
        print(cmd)
        print("speed",speed)
        serial.process_command(cmd,speed)

if __name__ == "__main__":
    serial=My_Serial(serial_name,115200)
    serial.process_forward()
