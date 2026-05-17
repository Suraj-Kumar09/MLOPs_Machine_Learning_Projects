import logging
import os
from datetime import datetime

# Create log filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Logs folder inside current project
log_dir = "logs"

# Create full path
logs_path = os.path.join(log_dir, LOG_FILE)

# Create logs folder
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=logs_path,
    format="[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG,
)