# This will train the bot
import logging
import fasttext

# Set Logging -- basic configuration

logging.basicConfig(format='%(asctime)s -- %(levelname)s: %(message)s',
                    level=logging.NOTSET,
                    datefmt='%Y-%m-%d %H:%M:%S')
my_logger = logging.getLogger('classifier_project')

# Train the regular model - BOT
my_logger.info("Training the regular BOT model ...")
model = fasttext.train_supervised(input="./cleansed_data/regular_input.txt", dim=300, epoch=10, minCount=5, loss='ns')

my_logger.info("Model saved into ./models/model_regular_BOT.bin")
model.save_model("./models/model_regular_BOT.bin")

my_logger.info("Model Testing Results - BOT Regular Codes:")
result = model.test("./cleansed_data/regular_input.txt")
my_logger.info(result)

# Train the rolled model - BOT
my_logger.info("Training the rolled BOT model ...")
model2 = fasttext.train_supervised(input="./cleansed_data/rolled_input.txt", dim=300, epoch=10, minCount=5, loss='ns')

my_logger.info("Model saved into ./models/model_rolled_BOT.bin")
model2.save_model("./models/model_rolled_BOT.bin")

my_logger.info("Model Testing Results - BOT Rolled Codes:")
result2 = model2.test("./cleansed_data/rolled_input.txt")
my_logger.info(result2)
