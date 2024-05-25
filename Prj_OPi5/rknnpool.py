from queue import Queue
from rknnlite.api import RKNNLite
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
 
 
def initRKNN(rknnModel="./weights/best_anchors.rknn", id=0):
    rknn = RKNNLite()
    #加载RKNN模型
    ret = rknn.load_rknn(rknnModel)
    if ret != 0:
        print("Load RKNN rknnModel failed")
        exit(ret)
    # 初始化 runtime 环境
    if id == 0:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)
    elif id == 1:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    elif id == 2:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_2)
    elif id == -1:
        ret = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    else:
        ret = rknn.init_runtime()
    if ret != 0:
        print("Init runtime environment failed!")
        exit(ret)
    print(rknnModel, "\t\tdone")
    return rknn
 
 
def initRKNNs(rknnModel="./weights/best_anchors.rknn", TPEs=1):
    rknn_list = []
    for i in range(TPEs):
        rknn_list.append(initRKNN(rknnModel, i % 3))
    return rknn_list
 
 
class rknnPoolExecutor():
    def __init__(self, rknnModel, img_size, anchors, TPEs, func):
        self.TPEs = TPEs
        self.queue = Queue()
        self.img_size=img_size
        self.anchors=anchors
        self.rknnPoollist = initRKNNs(rknnModel, TPEs)
        self.pool = ThreadPoolExecutor(max_workers=TPEs)
        self.lock=Lock()
        self.func = func
        self.num = 0
 
    def put(self, frame):
        with self.lock:
            self.queue.put(self.pool.submit(self.func, 
						self.rknnPoollist[self.num], self.img_size, self.anchors, frame))
            self.num += 1
        if self.num == self.TPEs:
            self.num = 0
 
    def get(self):
        if self.queue.empty():
            return None, False
        # 获取并等待第一个完成的任务
        future = self.queue.get()
        boxes, classes, scores = future.result()
        print("get success")
        
        return boxes, classes, scores, True

    def clear(self):
        with self.lock:
            while not self.queue.empty():
                future = self.queue.get_nowait()
                if not future.cancelled():
                    future.cancel()
                    
    def release(self):
        self.pool.shutdown()
        for rknn in self.rknnPoollist:
            rknn.release()
