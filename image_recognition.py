from imageai.Classification.Custom import CustomImageClassification
import os

execution_path = os.getcwd()

prediction = CustomImageClassification()
prediction.setModelTypeAsResNet50()

#set ModelPath to desired model. Copy the filename of the model and paste it after /models/
prediction.setModelPath("cannabis_checker/models/model_ex-008_acc-0.871136.h5")
prediction.setJsonPath("cannabis_checker/json/model_class.json")
prediction.loadModel(num_objects=6)


directory_to_test = "cannabis_checker/test/not_cannabis"

directory = os.fsencode(directory_to_test)
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".jfif") or filename.endswith(".png"): 
        predictions, probabilities = prediction.predictImage(f"{directory_to_test}/{filename}", result_count=3)

        for eachPrediction, eachProbability in zip(predictions, probabilities):
            print(eachPrediction , " : " , eachProbability)

        continue
    else:
        continue

