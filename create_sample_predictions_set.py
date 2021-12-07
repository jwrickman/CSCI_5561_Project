import json



with open("/media/storage2/open_monkey/monkey_train_annotations.json", "r") as fh:
    annotations = json.load(fh)["data"]

annotations = annotations[:100]



fake_predictions = []
for idx in range(len(annotations)):
    prediction = annotations[idx]
    landmarks = np.array(prediction["landmarks"])
    landmarks += np.random.randint(-50, 50, landmarks.shape)
    prediction["landmarks"] = list(landmarks)
    fake_predictions.append(prediction)


with open("fake_predictions.json", "w") as fh:
    json.dump(fake_predictions, fh)
