import torch
import json


def get_delta(out):
    prediction = np.unravel_index(np.argmax(out, axis=None), out.shape)
    delta_x = prediction[0] - 64
    delta_y = prediction[1] - 64
    return (delta_x, delta_y)

### Get Initial Predictions ###

## This should be in a csv file of the same format as the challenge download



predictions_file = "open_monkey_predictions.csv"

refined_predictions_file = "open_monkey_refined_predictions.csv"

for bodypart in [2]:
    print("Starting Refining: ", bodyparts[bodypart])
    dataset = OpenMonkeyChallengeCropDataset(
        annotations=annotations,
        image_path=Path("/media/storage2/open_monkey/train"),
        bodypart_idx=bodypart,
        crop_size=128,
        test_mode=True
    )

    checkpoint = 'bodypart_{}_best.ckpt'.format(bodypart)
    model = UNet_Lightning.load_from_checkpoint(checkpoint)
    for idx in range(len(dataset)):
        image = dataset.get_crop(idx)
        out = model(image)
        delta_x, delta_y = get_delta(out)
        annotations[idx][landmarks][bodypart * 2] += delta_x
        annotations[idx][landmarks][bodypart * 2 + 1] += delta_y
        annotations[idx][corrections].append(delta_x)
        annotations[idx][corrections].append(delta_y)

    with open("save_corrections.json", "w") as fh:
        json.dump(annotations, fh)

