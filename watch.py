import os
from datetime import datetime
import time
import re
# excute each n seconds
def timer(n):
    while True:
        try:
            time_now = time.strftime("%H:%M:%S", time.localtime())
            """ Nvidia Command """
            command = "nvidia-smi"
            msg = [re.findall(r'\d+', process) for process in os.popen(command) if "MiB" in process]
            nvidia_msg = msg[:2]
            process_msg = msg[2:]
            if "07:31:00" <= time_now <= "23:59:59" or "00:00:00" <= time_now <= "01:00:00":
                """ Fuser Command """
                for gpu in nvidia_msg:
                    if int(gpu[-1]) > 50:
                        for k in process_msg:
                            if int(k[-1]) > 1000:
                                command = "echo 'gaoya123' | sudo -S kill -19 " + k[1]
                                os.system(command)
                                print(k[1])
                print(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ":stop")
        except:
            pass
        time.sleep(n)
# 5s
timer(10)
