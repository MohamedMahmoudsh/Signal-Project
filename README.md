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

Explain the preprocessing steps performed on the raw ECG data to prepare it for model training.



## CNN model



## LSTM model



## Results
