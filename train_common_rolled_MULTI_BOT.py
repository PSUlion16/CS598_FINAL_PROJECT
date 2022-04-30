# This will train the bot
import logging
import fasttext

# Set Logging -- basic configuration

logging.basicConfig(format='%(asctime)s -- %(levelname)s: %(message)s',
                    level=logging.NOTSET,
                    datefmt='%Y-%m-%d %H:%M:%S')
my_logger = logging.getLogger('classifier_project')

# Train the regular model - BOT
my_logger.info("Training the rolled common MULTI BOT model ...")
model = fasttext.train_supervised(input="./cleansed_data/rolled_common_input.train", dim=300, epoch=25, lr=.05, minCount=5, loss='ova')

my_logger.info("Model saved into ./models/model_rolled_common_MULTI_BOT.bin")
model.save_model("./models/model_rolled_common_MULTI_BOT.bin")

my_logger.info("Model Testing Results - MULTI BOT Rolled Common Codes:")
result = model.test("./cleansed_data/rolled_common_input.test")
my_logger.info(result)