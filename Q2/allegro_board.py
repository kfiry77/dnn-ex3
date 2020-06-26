from tensorflow.keras import callbacks
from trains import Logger


class TrainsReporter(callbacks.Callback):
    def __init__(self)
        super(TrainsReporter, self).__init__()
        self.epoch_ref = 0

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        print("Stop training; got log keys: {}".format(keys))

    def on_epoch_end(self, epoch, logs=None):
        Logger.current_logger().report_scalar("loss",     "train",      iteration=epoch_ref + epoch, value=logs["loss"])
        Logger.current_logger().report_scalar("loss",     "validation", iteration=epoch_ref + epoch, value=logs["val_loss"])
        Logger.current_logger().report_scalar("accuracy", "train",      iteration=epoch_ref + epoch, value=logs["binary_accuracy"])
        Logger.current_logger().report_scalar("accuracy", "validation", iteration=epoch_ref + epoch, value=logs["val_binary_accuracy"])
