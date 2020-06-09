import os
import time
from collections import defaultdict, deque
from multiprocessing import Process, Queue
from threading import Lock, Thread
import random

import logging
_log = logging.getLogger(__name__)


class FPSMeter():
    def __init__(self, name='FPS', period=10.0):
        self.name = 'FPS for %s(PID-%d)' % (name, os.getpid())
        self.period = period
        self.last_time = time.time()
        self.count = 0.0
        self.total_count = 0
        self.dead_time = 0.0
        self.dead_time_clock = time.time()
        _log.info('Initilized counter %s' % self.name)

    def increment(self, steps=1, before_str=''):
        self.count += steps
        self.total_count += steps
        current_time = time.time()
        if current_time - self.last_time > self.period:
            s = before_str + '%s is %0.2f' % (self.name, self.count /
                                              (current_time - self.last_time))
            if self.dead_time > 0:
                s += '(with %0.1f%% down time)' % (min(100, 100.0 * (
                    self.dead_time / (current_time - self.last_time))))
            #_log.info(s)
            self.count = 0
            self.last_time = current_time
            self.dead_time = 0
            self.dead_time_clock = current_time
            return True
        else:
            return False

    def now_passive(self):
        self.dead_time_clock = time.time()

    def now_active(self):
        self.dead_time += time.time() - self.dead_time_clock


class ThreadSafeCounter():
    def __init__(self):
        self.lock = Lock()
        self.counter = -1

    def increment(self):
        with self.lock:
            self.counter += 1
            return self.counter


def worker(q_in, q_out, SlaveClass, params):
    try:
        slave = SlaveClass(*params)    
        while True:
            job_id, metadata, data = q_in.get()
            if not (data is None and metadata is None):
                res = slave.process(data)
            else:
                res = None
            q_out.put((job_id, metadata, res))
    except:
        import traceback; traceback.print_exc()
        q_out.put((None, None, None))  # Instant pison pill
        print('Instant poison pill sent...')
        os._exit(1)
        return


def result_sync_worker(q_in, q_out, require_order, task_name):
    wavefront = -1
    stack = dict()
    meter = FPSMeter(task_name, period=10)
    while True:
        job_id, metadata, res = q_in.get()
        if job_id is None: # Instant poison pill
            print('Instant poison pill recieved...')
            q_out.put((None, None))
            return        
        if require_order:
            stack[job_id] = (metadata, res)
            while (wavefront + 1) in stack:
                wavefront = wavefront + 1
                q_out.put(stack[wavefront])
                meter.increment()
                #print(wavefront)
                del stack[wavefront]
        else:
            if not (res is None and metadata is None):
                wavefront = job_id
                q_out.put((metadata, res))
                meter.increment()
            else: # need to run all remaining requests before halting
                last_job_id = job_id - 1
                while wavefront < last_job_id:
                    wavefront, metadata, res = q_in.get()
                    print(wavefront, last_job_id)
                    q_out.put((metadata, res))
                q_out.put((None, None))
                        

def poll_dq(dq):
    while True:
        try:
            return dq.popleft()
        except IndexError:
            time.sleep(0.01)


def input_worker(procs, dq_in, counter):
    prev_data = poll_dq(dq_in)
    #print('----------',len(dq_in))
    while True:
        for proc_ind in procs:
            if procs[proc_ind]['queue'].empty():
                #print('----------',len(dq_in))
                procs[proc_ind]['queue'].put((counter.increment(),
                                              prev_data[0], prev_data[1]))
                #print(counter.counter)
                prev_data = poll_dq(dq_in)


class JobDistributor():
    def __init__(self,
                 n_procs,
                 gpu_list,
                 SlaveClass,
                 params=(),
                 queue_size=None,
                 require_order=True,
                 task_name = 'JobDistributor'):
        assert require_order==True, 'Orderless operation is currently buggy and reaches premature halts'
        self.q_out_synced = Queue()
        self.n_procs = n_procs
        self.dq_in = deque(maxlen=queue_size)
        self.q_out = Queue()
        self.procs = defaultdict(dict)
        self.job_counter = ThreadSafeCounter()
        self.total_pushed_jobs = 0

        if self.n_procs > 0:
            for ii in range(int(self.n_procs)):
                self.procs[ii]['queue'] = Queue(maxsize=1)
                if gpu_list is not None:
                    time.sleep(1)
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_list[ii % len(gpu_list)])
                p = Process(
                    target=worker,
                    args=(self.procs[ii]['queue'], self.q_out, SlaveClass, params),
                    daemon=True)
                p.start()
                self.procs[ii]['proc'] = p
                #time.sleep(np.random.randint(100)/1000)

            t = Thread(
                target=input_worker,
                args=(self.procs, self.dq_in, self.job_counter),
                daemon=True)
            t.start()

            sync_worker = Process(
                target=result_sync_worker, args=(self.q_out, self.q_out_synced, require_order, task_name))
            sync_worker.daemon = True
            sync_worker.start()
        else:
            self.single_slave = SlaveClass(*params) 
            
    def push(self, data, metadata=[]):
        self.total_pushed_jobs += 1
        self.dq_in.append((metadata, data))

    def push_multi(self, datas):
        for data in datas:
            self.push(data)

    def process_all(self, metadatas_and_datas):
        if len(metadatas_and_datas) == 0:
            return []
        res = []
        if self.n_procs > 0:
            while self.q_out_synced.qsize() > 0:
                print('Job queue not embpty...')
                self.q_out_synced.get()
            for cnt, (metadata, data) in enumerate(metadatas_and_datas):
                self.push(data, metadata)
            #if 'cnt' not in locals():
            #    import ipdb; ipdb.set_trace()
            while cnt >= 0:
                res.append(self.q_out_synced.get())
                cnt -= 1
        else:
            for metadata, data in metadatas_and_datas:
                if data is not None:
                    res.append((metadata, self.single_slave.process(data)))
                else:
                    res.append(metadata, None)
        return res

    def process_all_no_metadata(self, datas):
        metadatas_and_datas = [([], data) for data in datas]
        return [res[1] for res in self.process_all(metadatas_and_datas)]

    
    def push_poison_pill(self):
        self.push(None, None)

    def qout_size(self):
        if self.n_procs > 0:
            return self.q_out_synced.qsize()
        else:
            return self.dq_in.qsize()

    def get(self):
        if self.n_procs > 0:
            return self.q_out_synced.get()
        else:
            metadata, data = self.dq_in.popleft()
            if data is not None:
                res = self.single_slave.process(data)
            else:
                res = None
            return (metadata, res)
        
    def __iter__(self):
        return self

    def __len__(self):
        return self.total_pushed_jobs

    def __next__(self):
        metadata, res = self.get()
        if res is not None:
            return metadata, res
        else:
            raise StopIteration()


if __name__ == '__main__':
    # Happy usage example
    import numpy as np

    class Friend():
        def __init__(self):
            pass

        def process(self, data):
            time.sleep(random.randint(0, 10) / 30.0)  # processing time
            #import ipdb; ipdb.set_trace()  # Check debud capability in non-distributed mode
            if data == 30: m = n # Check exeption handling
            return data

    distributor = JobDistributor(10, None, Friend, queue_size=1000, require_order=True)
    for ii in range(100):
        distributor.push(ii)
        print('input: %d'%ii) 
        time.sleep(0.001)
    distributor.push_poison_pill()
    
    for metadata, result in distributor:
        print('output: %d'%result)

    print('Waiting for jobs to halt...')
