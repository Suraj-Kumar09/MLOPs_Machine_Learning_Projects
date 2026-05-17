from credit_card.logger import logging
from credit_card.exception import CreditCardException
import sys

logging.info("Welcome to credit card custom logger!")


try:
    a = 1/0
except Exception as e:
    raise CreditCardException(e, sys)