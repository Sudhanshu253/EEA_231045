# Hand-sign detection model

This repository contains the source code for a hand-sign detection model built using Python.

## How it works

1. Run the file `collect_imgs.py`. This will collect the images for our dataset and update the path `DATA_DIR` and put your `data` folder path here.

2. Run `create_dataset.py`. This program works on MediaPipe, which will create landmarks on our hands to help in sign detection, update the path `DATA_DIR` and put your `data` folder path here also.

3. Now, run `train_classifier.py`. This works on random forest model of ML. This will split the dataset into training and test data. Update the path below with the path to your `data.pickle` file.
  
   ```python
   with open('/Users/sudhanshusaroj/myvenv/data.pickle', 'rb') as f:
   data_dict = pickle.load(f)
   ```

4. Finally, run `inference_classifier.py` for sign detection. Update the path below with the path to your `model.p` file created by `train_classifier.py` run in 3.

   ```python
   # Load the trained model
   with open('/Users/sudhanshusaroj/myvenv/model.p', 'rb') as f:
   ```
