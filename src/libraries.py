# Description: This file contains all the libraries used in the project

import os
import glob
import pickle

import numpy as np
import pandas as pd
import random
import math

import datetime

import matplotlib.pyplot as plt
from IPython.display import display
import ipywidgets as widgets
from ipywidgets import interact

from sklearn.metrics.pairwise import cosine_similarity

from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import SVD, accuracy
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
