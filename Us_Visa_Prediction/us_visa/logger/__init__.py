import logging
import os
from from_root import from_root
from datetime import datetime

# 1. Log file ka naam
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2. Log folder ka naam
log_dir = 'logs'

# 3. logs_path ko create karein (Sahi tarika)
# Hum logs folder ko project ke root folder ke andar banayenge
logs_path = os.path.join(os.getcwd(), log_dir, LOG_FILE)

# 4. Sirf folder ka path nikalne ke liye
logs_dir_path = os.path.join(os.getcwd(), log_dir)

# 5. Folder create karein agar nahi hai toh
os.makedirs(logs_dir_path, exist_ok=True)

# 6. Logging configuration
logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO, # Debug se INFO kar diya hai production ke liye
)