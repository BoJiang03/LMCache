import multiprocessing as mp, time, os
def child():
    t=time.time()
    while time.time()-t < 6:
        sum(i*i for i in range(10000))
if __name__ == "__main__":
    mp.set_start_method("spawn")
    p = mp.Process(target=child); p.start()
    print("parent", os.getpid(), "child", p.pid, flush=True)
    p.join()
