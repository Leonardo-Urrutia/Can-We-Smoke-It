from imageai.Classification.Custom import CustomImageClassification
import os

execution_path = os.getcwd()

prediction = CustomImageClassification()
prediction.setModelTypeAsResNet50()
prediction.setModelPath("cannabis_checker/models/model_ex-008_acc-0.871136.h5")
prediction.setJsonPath("cannabis_checker/json/model_class.json")
prediction.loadModel(num_objects=2)

# for filename in os.listdir(directory):
#     if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".jpeg"): 
         
#         continue
#     else:
#         continue


directory_in_str = "cannabis_checker/test/not_cannabis"

directory = os.fsencode(directory_in_str)
    
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg") or filename.endswith(".jpeg"): 
        predictions, probabilities = prediction.predictImage(f"{directory_in_str}/{filename}", result_count=1)

        for eachPrediction, eachProbability in zip(predictions, probabilities):
            print(eachPrediction , " : " , eachProbability)

        continue
    else:
        continue


# for eachPrediction, eachProbability in zip(predictions, probabilities):
#     print(eachPrediction , " : " , eachProbability)