# import sys
# from us_visa.logger import logging
# from us_visa.exception import USvisaException

# try:
#     logging.info("Testing custom exception and logger")
#     # Jan-boojh kar division by zero error create kar rahe hain test karne ke liye
#     result = 10 / 0 
# except Exception as e:
#     logging.info("Error occurred, raising custom exception")
#     raise USvisaException(e, sys)


## Mongodb url ko env me store krne ke bad ka cheak karenge

# import os
# mongodb_url = os.getenv('MONGODB_URL')
# print(mongodb_url)



## Pipelines banane ke bas cheak kr rahe hai 

from us_visa.pipeline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()
