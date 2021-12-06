import torch








def get_delta(out):
    prediction = np.unravel_index(np.argmax(out, axis=None), out.shape)
    delta_x = prediction[0] - 64
    delta_y = prediction[1] - 64
    return (delta_x, delta_y)




### Get Initial Predictions ###

## This should be in a csv file of the same format as the challenge download



predict_file = "open_monkey_predictions.csv"

refined_predictions = "open_monkey_refined_predictions.csv"

for bodypart in range(13):
    dataset = OpenMonkeyChallengeCropDataset(
        annotations=annotations,
        image_path=Path("/media/storage2/open_monkey/train"),
        bodypart_idx=bodypart,
        crop_size=128,
        test_mode=True
    )

    model = # Load correct bodypart model

    for idx in range(len(dataset)):
        image = dataset.get_crop(idx)
        ## Preprocess image??? ##
        out = model(image)



# For body part in parts list
## Crop to body part
## Run corresponding predictor
## Update Guess
