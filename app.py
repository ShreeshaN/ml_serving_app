from flask import Flask

app = Flask(__name__)


@app.route('/')
def welcome():
    return "Welcome to Pluralsight's machine learning tutorials. \n" \
           "In this tutorial you will learn a few things:\n" \
           "\t1. How to build a simple machine learning classifier (We can eventually move towards complex architectures based on data)\n" \
           "\t2. How to serve the model once it is trained (i.e How to use it in production)\n" \
           "\t3. How to maintain it and version it to fit future updates"


if __name__ == '__main__':
    app.run()
