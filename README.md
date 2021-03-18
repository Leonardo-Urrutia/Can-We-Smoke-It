# Can-We-Smoke-It
Utilizing machine learning to build the brain for an app to allow users to input images of plants and have it determine if it is cannabis or not cannabis.

We all researched different models for image classification / image identification.
Calvin found ImageAI and proposed this model for the team.
We then all used an instagram scrapper to gather various cannabis images and found datasets for noncannabis.
Leo attempted to utilize S3 to store our images, but we hit a pay wall.
We then pivoted and aggregated our images in google drive.
We then ran our model, and hit a few snags, causing us to run the model in two different ways.
1. Training model using all our images and adjusting our hyper parameters (Leo) 
2. Training model to our images one catagory/classification at a time (Calvin / Tyson)
We then identified the model/epoch that showed the most functionality (actually identifying cannabis vs noncannabis)
We then created a roadmap of how we would improve this model using annotations in PascalVOC format, and freezing our layers doing the training.


