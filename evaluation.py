import numpy as np
import h5py
import json




ANNOTATION_FILE = "/media/storage2/open_monkey/monkey_val_annotations.json"
H5_FILE = "/media/storage2/open_monkey/val_data.h5"
MODEL_FILE = "/media/storage2/open_monkey/checkpoints/Best_LEAP.h5"
LOG_FILE = "/media/storage2/open_monkey/log_deep.h5"

def MPJPE(y_error):
    result = np.sqrt(np.sum(y_error ** 2, axis=2))
    result = np.mean(result, axis=0)
    result = result / 128
    return np.mean(result)


def PCK(y_error, err):
    result = np.sqrt(np.sum(y_error ** 2, axis=2))
    result = (result / 128) <  err
    result = np.mean(result, axis=1)
    return np.mean(result)


if __name__ == "__main__":

    with h5py.File(LOG_FILE, "r") as results:
        results = np.array(results["logs"]["y_error"])[1]
        print(MPJPE(results))
        print(PCK(results, 0.2))
        print(PCK(results, 0.5))


#    ### Load Data ###
#    data_generator = DataGenerator(H5_FILE)
#    train_generator = TrainingGenerator(generator=data_generator,
#                                downsample_factor=0,
#                                augmenter=augmenter,
#                                sigma=5,
#                                validation_split=0.1,
#                                use_graph=True,
#                                random_seed=2,
#                                graph_scale=1)
#
#
#    ### Load Model ###
#    model = LEAP(train_generator)
#
#    for X, y in data_generator:
#        out = model.predict(X, batch_size=1)
