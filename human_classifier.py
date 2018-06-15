import numpy as np
import json
from model import VideoLearn
# load features and tags
features_file="features.txt"
tags_file="tags.npy"
tags = np.load(tags_file)
with open(features_file) as f:
    features = json.load(f)
learning = VideoLearn(4, 5, 0.001)
learning.learn(features, tags, 10)



