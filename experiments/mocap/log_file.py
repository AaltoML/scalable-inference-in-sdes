import logging
import os


class LogFile:
    def __init__(self, file_path):
        if file_path is None:
            self.create_log_file = False
        else:
            self.create_log_file = True
            if not os.path.exists(file_path):
                os.makedirs(file_path)

            logger = logging.getLogger()
            if len(logger.handlers) > 0:
                logger.handlers[0].stream.close()
                logger.removeHandler(logger.handlers[0])

            self.file_path = os.path.join(file_path, "log.out")

            logging.basicConfig(
                filename=self.file_path,
                level="INFO",
                format="%(asctime)s - %(message)s",
            )

    def log(self, val):
        if self.create_log_file:
            logging.info(val)

    def get_previous_epochs(self):
        with open(self.file_path) as f:
            data = f.readlines()

        data.reverse()
        epoch_val = 0
        for d in data:
            if "Epoch" in d:
                epoch_val = int(d.split("Epoch")[1].split(":")[0])
                break
        return epoch_val
