# CodeALpha_Music_Generatoin_with_AI
Music_Generatoin_with_AI
##########################################################################################################################################################################
Task-1 Music Generation with AI :

Excited to share my latest Project with @Code Alpha …!

Problem Statement:
Create an AI-powered music generation system capable of composing original music. Utilize deep learning techniques like Recurrent Neural Networks (RNNs) or Generative Adversarial Networks (GANs) to generate music sequences.

Description:
As a part of my internship offered by CodeAlpha I have completed my first task in which I have made a AI based Music Generator generate which creates new MIDI music sequences by training an LSTM-based Recurrent neural network(RNN) on MIDI files. It uses deep learning to understand patterns in musical notes, allowing the generation of new sequences based on these learned patterns.

Working:
It initially read the input MIDI files and extracts note sequences and stores them in a format suitable for model training.Then it normalises the note data using Min-max scaling.Then after it implements Sequential neural network with LSTM layers, which are well-suited for handling time series data like music sequences.This model saves the model checkpoints during training, allowing training to be resumed or results to be saved for each epoch which avoid having to retrain the model from scratch every time you need to use it. Finally after training it generates new sequences of notes and saves them as a new MIDI file. 

Technologies Used:
Python as programming language used to implement this code.
TensorFlow and Keras are used for building and training the recurrent neural network model. 
Mido is a Python library for handling MIDI file operations, such as reading, writing, and modifying MIDI tracks.
Scikit-learn used to preprocessing the data and using utilities such as Min-Max scaling and normalizing the note data.
LSTM (Long Short-Term Memory): A type of Recurrent Neural Network (RNN) that captures long-range dependencies, essential for generating coherent music sequences.

It involves Machine Learning Pipeline processes such as Data preprocessing, Feature Engineering, Splitting of data, Model Building, Model Training, Prediction of data, Generation of output.
