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

### First we need to undestand what is the sinus Bradycardia
Bradycardia is a slower than normal heart rate. A normal adult resting heart rate is between 60 – 100 beats per minute (bpm). If you have bradycardia, your heart beats fewer than 60 times a minute.
![download](https://github.com/MohamedMahmoudsh/Signal-Project/assets/113555799/6dc14a2d-056b-4f6f-9899-6e5002492f98)

### Our finding about preprocessing

1 - We find that data contains 60 disease so we need to filter only a Normal(Sinus Rythme) and abnormal case (Siuns Bradycardia).
2 - We found that many pearson can have more than one disease.
3 - In signal processing domain we can store actual data in (.mat) file and store the data description in (.hea) file. 
ex patient1.mat , patient1.hea 
4 - We have data description (disease name encoded with numbre) in seperate file . 
### How can adress this problem  ? 
1 - Store your data directory(.hea file , .mat file ) in list  
2 - filter data by choosing .hea files only
3 - Open .hea file (contain labels)  and extract the labels
4 - Create a Dataframe and store the file name (but without directory ex patient1) ,and the label
    since we have more than one disease so  we store the the label info inside a list
5 - Choose only the recoerd which contain a length one array 

## CNN model
  The convolutional Neural Network (CNN) model is designed for binary classification task between normal sinus rhythm and Sinus Bradycardia  **which consists of the following layers**
  * **Input Layer**: The model starts with a 1D convolutional layer (Conv1D) consisting of 128 filters, each with a kernel size of 55 and ReLU activation. This layer is tailored to process input signals of length 5000 with a single feature.
  * **Pooling layer** : Following the convolutional layer, a max-pooling layer (MaxPooling1D) is applied with a pool size of 10.
  *  **Dropout** : is then introduced with a rate of 0.5 to mitigate overfitting
  *  **Global Average Pooling**: is employed to condense the feature maps into a single vector, facilitating the transition to the fully connected layers.
  *  **Dense layer** : with ReLU activation, gradually reducing the dimensionality while extracting high-level features, Each dense layer is followed by dropout regularization to further prevent overfitting the final layer  with a sigmoid activation function is utilized to produce the binary classification output.
    


## LSTM model



## Results
