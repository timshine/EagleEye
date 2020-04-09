import psutil
import shutil
import platform
from datetime import datetime

def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

# Get system info
def get_sys_info():
    sys_info = dict()
    uname = platform.uname()
    sys_info['Operating System'] = uname.system
    sys_info['Computer Name'] = uname.node
    sys_info['Version'] = uname.version
    sys_info['Machine'] = uname.machine
    sys_info['Processor'] = uname.processor
    return sys_info



# Boot Time
def get_boot_time():
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    last_boot = (f"{bt.month}/{bt.day}/{bt.year} {bt.hour}:{bt.minute}:{bt.second}")
    return last_boot


#CPU Info
def get_cpu_cores():
    cpu_info = dict()
    cpu_info["Physical Cores"] = psutil.cpu_count(logical=False)
    cpu_info["Total Cores"] = psutil.cpu_count(logical=True)
    # CPU usage
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True)):
        cpu_info[f"Core {i}"] = f"{percentage}%"
    cpu_info["Total CPU Usage"] = f"{psutil.cpu_percent()}%"
    return cpu_info 


# Memory Information
def get_memory_info():
    # RAM details
    memory_info = dict()
    svmem = psutil.virtual_memory()
    memory_info["Total RAM"] = get_size(svmem.total)
    memory_info["RAM Available"] = get_size(svmem.available)
    memory_info["RAM Used"] = get_size(svmem.used)
    memory_info["RAM Percentage"] = f"{svmem.percent}%"

    # Disk Information
    total, used, free = shutil.disk_usage("/")
    memory_info["Disk Size"] = get_size(total)
    memory_info["Disk Used"] = get_size(used)
    memory_info["Disk Free"] = get_size(free)
        
    return memory_info
