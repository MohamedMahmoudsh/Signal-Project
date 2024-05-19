# ECG Signal Classification

A project focused on classifying ECG signals using deep learning techniques.

## Description

This project is dedicated to the classification of ECG (Electrocardiogram) signals through the application of deep learning models. By leveraging advanced neural network architectures, the aim is to accurately identify different types of heart conditions from ECG data. This tool can assist in the early detection and diagnosis of cardiac anomalies, potentially saving lives by enabling timely medical intervention.



## Table of Contents

1. [Data Description](#data-description)
2. [Preprocessing Steps](#preprocessing-steps)
3. [CNN Model](#cnn-model)
4. [LSTM Model](#lstm-model)
5. [Results](#results)

## Data Description

#### General information about the data
-First, each subject underwent a 12-lead resting ECG test that was taken over a period of 10 seconds.<br>
-This database consists of 45,152 patient ECGs. The number of volts per A/D bit is 4.88, and A/D converter had 32-bit resolution. The amplitude unit was microvolt. The upper limit was 32,767, and the lower limit was −32,768.<br>
-There are labels of each subject’s rhythm and other conditions such as PVC, right bundle branch block (RBBB), left bundle branch block (LBBB), and atrial premature beat (APB).<br>
-All recordings are organized in two levels folder directory under the WFDBRecords folder.<br>

<h3 style="color:green;"> The main problem we focused on </h3>
Arrhythmia constitutes a problem with the rate or rhythm of the heartbeat, and an early diagnosis is essential for the timely inception of successful treatment.<be>

#### Types of diseases that can cause arrhythmia.
-atrial fibrillation.<br>
-general supraventricular.<br>
-sinus bradycardia.<br>
-sinus irregularity rhythm.<br>
#### The main disease that was chosen from this dataset
-**sinus bradycardia:** is a condition characterized by an abnormally slow heart rate originating from the sinus node, which is the natural pacemaker of the heart.<br>
-In adults, sinus bradycardia is typically defined as a resting heart rate of fewer than 60 beats per minute.<br>

- **Source**:
  - [PhysioNet](https://physionet.org/content/ecg-arrhythmia/1.0.0/WFDBRecords/01/#files-panel) -That the link of the data on PhysioNet.
  - [Kaggle](https://www.kaggle.com/datasets/erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database) -The sample of data we use on Kaggle.
  - [Scientific Data](https://doi.org/10.1038/s41597-020-0386-x) -The paper which contains details about the data.

- **Format**: Description of file formats Each recorder consists of a header file (.hea) and a data file (.mat).
  
- **Characteristics**: Number of samples:45,152 , duration of ECG signals:10 seconds , 500 Hz sampling rate.

## Preprocessing Steps
In our data we have alot of disease , but we need to focus on Normal case and Sinus Bradycardia.

![download](https://github.com/MohamedMahmoudsh/Signal-Project/assets/113555799/6dc14a2d-056b-4f6f-9899-6e5002492f98)

### Our finding about preprocessing

- We find that data contains 60 disease so we need to filter only a Normal(Sinus Rythme) and abnormal case (Siuns Bradycardia).
- We found that many pearson can have more than one disease.
- In signal processing domain we can store actual data in (.mat) file and store the data description in (.hea) file. 
-ex patient1.mat , patient1.hea 
- We have  disease name encoded with numbre in seperate csv file named by "ConditionNames_SNOMED-CT.csv".

### How can adress this problem  ? 

- Store your data directory(.hea file , .mat file ) in list .
- filter data by choosing .hea files only.
- Open .hea file (contain labels)  and extract the labels.
- Create a Dataframe and store the file name (but without directory ex patient1) ,and the label.
-   since we have more than one disease so  we store the the label info inside a list.
- Choose only the recoerd which contain a length one array.
- Search for Sinus Rhythm and Sinus Bradycardia Id.
- Filter only the labels with the given Id.
- Save this portion of data in csv.
- Encode "SR" for Normal , "SB" for abnormal
- We will use only one channel lead I channel 


![image](https://github.com/MohamedMahmoudsh/Signal-Project/assets/113555799/bd11f4df-c9bd-4f31-844d-ad9088dc7614)

### How to use this csv ?
-Just by reading csv read all stored directory and store it a numpy array.

-Decod the label zero for Normal , one for Abnormal.

[Preprocessing Notebook](https://github.com/MohamedMahmoudsh/Signal-Project/tree/main/Preprocessing)
## CNN model
  The convolutional Neural Network (CNN) model is designed for binary classification task between normal sinus rhythm and Sinus Bradycardia  **which consists of the following layers**
  * **Input Layer**: The model starts with a 1D convolutional layer (Conv1D) consisting of 128 filters, each with a kernel size of 55 and ReLU activation. This layer is tailored to process input signals of length 5000 with a single feature.
  * **Pooling layer** : Following the convolutional layer, a max-pooling layer (MaxPooling1D) is applied with a pool size of 10.
  *  **Dropout** : is then introduced with a rate of 0.5 to mitigate overfitting
  *  **Global Average Pooling**: is employed to condense the feature maps into a single vector, facilitating the transition to the fully connected layers.
  *  **Dense layer** : with ReLU activation, gradually reducing the dimensionality while extracting high-level features, Each dense layer is followed by dropout regularization to further prevent overfitting the final layer  with a sigmoid activation function is utilized to produce the binary classification output.

```python
CNN_model = Sequential()
CNN_model.add(Conv1D(128, 55, activation='relu', input_shape=(5000, 1)))
CNN_model.add(MaxPooling1D(10))
CNN_model.add(Dropout(0.5))
CNN_model.add(Conv1D(128, 25, activation='relu'))
CNN_model.add(MaxPooling1D(5))
CNN_model.add(Dropout(0.5))
CNN_model.add(Conv1D(128, 10, activation='relu'))
CNN_model.add(MaxPooling1D(5))
CNN_model.add(Dropout(0.5))
CNN_model.add(Conv1D(128, 5, activation='relu'))
CNN_model.add(GlobalAveragePooling1D())
CNN_model.add(Dense(256, kernel_initializer='normal', activation='relu'))
CNN_model.add(Dropout(0.5))
CNN_model.add(Dense(128, kernel_initializer='normal', activation='relu'))
CNN_model.add(Dropout(0.5))
CNN_model.add(Dense(64, kernel_initializer='normal', activation='relu'))
CNN_model.add(Dropout(0.5))
CNN_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
CNN_model.compile(loss = 'binarycrossentropy')

```
  *  **early_stopping function** : is a technique used during training to prevent overfitting by stopping training when the model's performance on a validation dataset starts to degrade, is passed to the fit function as a part of the callbacks

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```
## LSTM model

The Long Short-Term Memory (LSTM) network architecture employed in this project is designed to analyze sequential data such as Electrocardiogram (ECG) signals. LSTMs are particularly effective in capturing long-term dependencies in time-series data, making them suitable for tasks like ECG classification.

#### Architecture

The LSTM model architecture is defined as follows:

 * **input Shape**:The input data has a shape of (100, 50), where 100 represents the number of time steps and 50 represents the number of window size at each time step.<br>
  * **LSTM Layer**: The LSTM layer consists of 64 units, allowing the model to learn temporal patterns within the data.<br>
 * **Dense Layers**: Two dense layers follow the LSTM layer, with 32 units in the first layer and 1 unit in the output layer. The activation functions used are ReLU and Sigmoid, respectively.<br>
 * **Compilation**: The model is compiled using the Adam optimizer and binary crossentropy loss function for binary classification tasks. The accuracy metric is used to monitor the model's performance during training.<br>

```python
optimizer = Adam()
recurrent_model = Sequential()
recurrent_model.add(Input(shape=(100, 50)))
recurrent_model.add(LSTM(64))
recurrent_model.add(Dense(32, activation='relu'))
recurrent_model.add(Dense(1, activation='sigmoid'))
recurrent_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
```
**Callback: Checkpoint**
In addition to the model architecture, a callback named checkpoint is incorporated during training. Checkpointing is a crucial technique in deep learning that allows the model's weights to be saved periodically, ensuring that the best-performing model is retained.<be>

```python
ModelCheckpoint(filepath=Best_Recurent_model_directory ,save_best_only=True)
```
<img src="https://github.com/MohamedMahmoudsh/Signal-Project/blob/main/Photos/LSTM.png" width="350" height="350">


# Results 
we best describe the performance of the model through **ROC Curve (Receiver Operating Characteristic Curve)** and **Confusion Matrix** 
* **The ROC curve**:  is a graphical plot that illustrates the diagnostic ability of a binary classifier as its discrimination threshold is varied where the plot has The y-axis representing the True Positive Rate (sensitivity), and the x-axis representing the False Positive Rate (1-specificity).
*  **Confusion Matrix**: This matrix provides a detailed breakdown of the classification performance by comparing the actual versus predicted classifications.
 ## for CNN Model  
 * In ROC due to high True Positive Rate (TPR) and low False Positive Rate which result in an AUC of 0.999812 so this indicates excellent performance for the model.
 *  IN the matrix we get 427 instances of True Positives(TP), 284 instances of True Negatives (TN), 3 instances of False Positives (FP), and 0 instances of False Negatives (FN).
 * accuracy of about 0.996
 * f1 score of about 0.996
 <div>  
<img src="https://github.com/MohamedMahmoudsh/Signal-Project/blob/main/Photos/Confusion%20matrix%20of%20CNN.png" width="350" height="350">
<img src="https://github.com/MohamedMahmoudsh/Signal-Project/blob/main/Photos/ROC%20OF%20CNN.png" width="350" height="350">
</div> 

 ## For LSTM Model
 
 * In ROC due to high True Positive Rate (TPR) and low False Positive Rate which result in an AUC of 0.9929 this indicates excellent performance for the model.
 * IN the matrix we get 418 instances of True Positives(TP), 278 instances of True Negatives (TN), 9 instances of False Positives (FP), and 9 instances of False Negatives (FN).
 * accuracy of about 0.975
 * f1 score of about 0.979
<div>  
<img src="https://github.com/MohamedMahmoudsh/Signal-Project/blob/main/Photos/Confusion%20marix%20of%20LSTM.png" width="350" height="350">
<img src="https://github.com/MohamedMahmoudsh/Signal-Project/blob/main/Photos/ROC%20OF%20LSTM.png" width="350" height="350">
</div> 
 
