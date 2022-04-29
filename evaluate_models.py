# This will evaluate the models!
import logging
import fasttext

# Set Logging -- basic configuration
logging.basicConfig(format='%(asctime)s -- %(levelname)s: %(message)s',
                    level=logging.NOTSET,
                    datefmt='%Y-%m-%d %H:%M:%S')
my_logger = logging.getLogger('classifier_project')

