import sys
from us_visa.logger import logging
from us_visa.exception import USvisaException

try:
    logging.info("Testing custom exception and logger")
    # Jan-boojh kar division by zero error create kar rahe hain test karne ke liye
    result = 10 / 0 
except Exception as e:
    logging.info("Error occurred, raising custom exception")
    raise USvisaException(e, sys)