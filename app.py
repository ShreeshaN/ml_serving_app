# -*- coding: utf-8 -*-
"""
@created on: 1/23/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description: Start point of this serving app. Run this file to host this app on a host machine. It will launch itself
                as a web server with multiple REST end points. Check documentation for REST end points.
                We are gonna have fun! Take a chill pill and keep going!

..todo::

"""

from flask import Flask
from pl_src.controllers import train_controller

app = Flask(__name__)
app.register_blueprint(train_controller)


@app.route('/')
def welcome():
    return "Welcome to Pluralsight's machine learning tutorials. \n" \
           "In this tutorial you will learn a few things:\n" \
           "\t1. How to build a simple machine learning classifier (We can eventually move towards complex architectures based on data)\n" \
           "\t2. How to serve the model once it is trained (i.e How to use it in production)\n" \
           "\t3. How to maintain it and version it to fit future updates"


if __name__ == '__main__':
    app.run()
