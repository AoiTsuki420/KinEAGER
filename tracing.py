import sys
import threading
import time

def trace_lines(frame, event, arg):
    if event != 'line':
        return
    trace_lines.current_frame = frame
    return trace_lines

def show_position():
    """后台线程：每秒输出当前执行位置"""
    while True:
        f = getattr(trace_lines, "current_frame", None)
        if f:
            filename = f.f_code.co_filename
            lineno = f.f_lineno
            funcname = f.f_code.co_name
            if "site-packages" not in filename and "lib/python" not in filename:
                print(f"[{time.strftime('%H:%M:%S')}] {filename}:{lineno} -> {funcname}", flush=True)
        time.sleep(1)

def start_tracing():
    """强制在主线程生效"""
    sys.settrace(trace_lines)
    threading.settrace(trace_lines)
    t = threading.Thread(target=show_position, daemon=True)
    t.start()
