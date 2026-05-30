import psutil
import threading
import os
import time
import csv
from typing import Callable

def sample_tree(stop, rows, interval=0.2, debug=False):
    pid = os.getpid()
    print(f"Starting monitoring process {pid}")
    parent = psutil.Process(pid)
    t0 = time.perf_counter()
    misses = 0
    while not stop.is_set():
        kids = parent.children(recursive=True)
        try: 
            parent_uss = parent.memory_full_info().uss
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            if debug:
                print(f"Failed to read parent process memory: {e}")
            parent_uss = 0
        kids_uss = 0
        for p in kids:
            try:
                kids_uss += p.memory_full_info().uss
            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                misses += 1
                if debug:
                    print(f"Failed to read child process memory: {e}")
                pass
        t = time.perf_counter() - t0
        rows.append((t,parent_uss/1e6,kids_uss/1e6,len(kids),misses))
        time.sleep(interval)

def run_with_monitor(
    function: Callable,
        *args,
        _interval: float = 0.2,
        _profile_output: str = "mem_trace.csv",
        **kwargs
    ):
    stop, rows = threading.Event(), []
    monitor = threading.Thread(target=sample_tree, args=(stop,rows,_interval), daemon=True)
    monitor.start()
    
    try:
        return function(*args,**kwargs)
    finally:
        stop.set(); monitor.join()
        
        with open(_profile_output,"w",newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_sec","parent_mb","workers_mb","n_workers","misses"])
            w.writerows(rows)
        
        peak = max((p+k for _,p,k,_,_ in rows),default=0)
        print(f"\nRun finished\n\tpeak total memory usage: {peak:.0f} MB")
        if rows:
            print(f"\ttotal time taken: {rows[-1][0]}\n")