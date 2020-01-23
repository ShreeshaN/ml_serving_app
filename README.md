# ml_serving_app
A simple application to serve machine learning models over the network using REST APIs 


### Prerequisites
```
1. Python3
2. scikit-learn 
3. Numpy
4. Pandas
```

### Data

```Iris``` dataset has been used in this project.
It is saved in 'data/' folder under the root of this project.

### How to run 

```
python3 app.py
```

The above command will host the app on http://127.0.0.1:5000. Please check it out as we have an encouraging message for all our viewers.


### How to train

Once the app is hosted. Hit the below URL
```http request
http://127.0.0.1:5000/train_decision_tree
```

### How to serve a trained model
After hitting the above URL, a decision tree is fit to IRIS data and is saved. 
It is now ready for evaluation and serving purpose.

Send a post request to the below URL with a JSON body containing the data for which prediction is required
```http request
http://127.0.0.1:5000/serve_decision_tree
Method: POST
Body: {"data":[0.1,1.1,0.4,0.8]}
Content Type: application/json 
```


### Run test cases
Find all he test cases under tests folder in the root of this project.
Run the below command to test our methods
```
python3 test_controllers.py
```



### Now, coming to the questions:

i. This dataset was obviously quite small, in the product you will be working with much
more data. How would you scale your training pipeline and/or model to handle datasets
which do not easily fit into system memory?


> There are two things to handle here:
> 1. Handling large datasets from the model training perspective
> 2. Handling large datasets over the network for training
>
> In the first case, where the dataset is so huge that it doesn't fit in the memory, we can use [Dataset API](https://www.tensorflow.org/api_docs/python/tf/data/Dataset) from tensorflow or [DataLoader API](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) from pytorch which do not load the entire data into memory but loads a subset of data required while training the minibatch. When it comes to loading a huge model into memory we could use distributed training using Horovod. I have given a detailed [talk](https://shreeshan.github.io/presentations) as to how we could implement distributed training in tensorflow. 
> In the second case, where data needs to be streamed for training to happen, then we could use data streaming and queuing platforms like Kafka/Flink/Spark Streaming to keep the training up as and when a batch of data is streamed


ii. Describe your optimal versioning strategy for APIs which expose machine learning
models. How does training the model on new data fit into versioning strategy? List the
pros and cons of your described strategy in detail.
> I would like to start by saying I have not spent extensive time doing this at work. But the way I would version my API's are based on new releases of trained models. Each serve API will have a version number on which it loads the models. Say for example a model is trained and saved under 'folder1' and is ready for serving. The serving API's URL would look like this: ip:/models/serve/decisiontree/folder1. Now when a new model is trained it would have a new folder under which it is saved and it should be reflected in the URL used for serving.


iii. Describe your choice of model and how it fits the problem. List benefits and drawbacks
of this type of model used in the way you have chosen and where there may be scaling
issues as a system like this grows in size or complexity. 

> I have used a Decision Tree for this problem. Since decision trees sketch out flowchars to arrive at a final solution, on the other hand our feature space contains just 4 features it would be a easy task for decision tree to come up with its sketch. However any simple classification algorithm or even a simple one layer Neural Network also will do the job equally well.<br>
> Advantages:<br>
> 1. Easy to implement and understand
> 2. Trains faster 
> 3. Does not over fit as the data and model for weight equal on complexity levels
>
> Disadvantages:<br>
> 1. Fails to work equally well on large datasets with millions of samples
> 2. Less number of parameters to fit high dimensional feature spaces
> 3. Does not work well when it comes to time series data
> 4. The pipeline written now does not scale well as no data streaming, distributed model training was implemented during development cycles. 
