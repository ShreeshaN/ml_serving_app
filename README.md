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
```
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

