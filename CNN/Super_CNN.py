import subprocess
import logging
import os
from datetime import datetime

actual_time = datetime.now()

userPath = os.getcwd().split('/')[2]
if userPath == "newfasant2":
    userPath = userPath + "/N101"

logs_folder_path=f'/home/{userPath}/N101-IA/CNN/SuperLogs'
os.makedirs(logs_folder_path, exist_ok=True)
logging.basicConfig(
    filename=f'/home/{userPath}/N101-IA/CNN/SuperLogs/Day_{actual_time.day}_{actual_time.month}_{actual_time.year}_Time_{actual_time.hour:02d}_{actual_time.minute:02d}_dataset.log',  # Name of the log file
    level=logging.INFO,  # Logs level (INFO, DEBUG, ERROR, etc.)
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format
)

list_epochs = ["13", "17", "18", "47"]
list_types = ["npy", "field", "ISAR", "field_ph_npy"]

for i in range(len(list_epochs)):
    for j in range(len(list_types)):
        result = subprocess.run(["python", "/home/newfasant2/N101/N101-IA/CNN/CNN_general.py", "-i","/home/newfasant2/N101/N101-IA/Datasets/Reorganized/Classification_1960_0_16_f_16_d_POV_5.16_SNR_5.0",
                         "-d", list_types[j], "-e", list_epochs[i]], capture_output=True, text=True)
        print(result.stdout,"\n")
        logging.info(result.stdout)