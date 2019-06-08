import pickle
import os

from multiprocessing import Process, Queue,Lock


class Logger(object):
    def __init__(self):
        self.log_data_dict = {}
        self.output_dir = None

    def setup(self,output_dir, multi_process = False):
        self.output_dir = output_dir
        self.multi_process = multi_process
        if self.multi_process:
            self.queue = Queue()
            self.lock = Lock()
            self.sub_process = Process(target = self.start_log,args = (self.queue,self.lock))
            self.sub_process.start()
            
    def get_dir(self):
        return self.output_dir
        
    def add_scalar(self,name,y,x):
        if name not in self.log_data_dict:
            self.log_data_dict[name] = []
        self.log_data_dict[name].append([y,x])
        
    def save_dict(self):
        with open(os.path.join(self.output_dir,'log_data_dict.pkl'),'wb') as f:
            pickle.dump(self.log_data_dict,f)
            
    def trigger_close(self):
        if self.multi_process:
            self.queue.put( ('__close__',0,0) ,block = True)
            
    def trigger_log(self,name,y,x):
        if self.multi_process:
            self.queue.put((name,y,x),block = True)
        else:
            self.add_scalar(name,y,x)
        
    def trigger_save(self):
        if self.multi_process:
            self.queue.put(('__save__',0,0),block = True)
        else:
            self.save_dict()
        
    def start_log(self,queue,lock):
        while True:
            item = queue.get(block = True)
            name,y,x = item
            if name =='__close__':
                break
            lock.acquire()
            if name == '__save__':
                self.save_dict()
            else:
                self.add_scalar(name,y,x)
            lock.release()
            
Singleton_logger = Logger()