# ROAD TRAFFIC SIGN DETECTION USING DEEP LEARNING
## Introduction
Across different nations, over speeding consistently ranks as a predominant cause of road accidents. The propensity for drivers to exceed prescribed speed limits poses a severe threat to road safety. The need for effective measures to curb over speeding and enforce speed limits underscored by its persistent association with road accidents. Another significant contributor to road accidents is the disregard or oversight of traffic signs. Whether due to driver negligence, obscured signage, or inadequate awareness, the failure to adhere to the information conveyed by traffic signs introduces an element of unpredictability on the roads. Missing or ignoring crucial signs, such as stop signs, speed limits, and warnings, undermines the overall efficacy of traffic management systems and increases the likelihood of collisions.

In this context, The primary goal of road safety initiatives is to establish an autonomous intelligent traffic system capable of seamlessly detecting and identifying traffic signs, subsequently alerting users about upcoming signs for enhanced awareness. Our initiative is to address this problem by developing a Model using deep learning techniques, that can detect the road traffic signs ahead and warn the riders. By addressing the root causes of accidents related to over speeding and ignored traffic signs, this project contributes to a safer and more efficient transportation ecosystem.

## Methodology
  The methodology employed in this project leverages the <b>ResNet (Residual Network)</b> architecture. To expedite the training process and enhance the model's adaptability to the specific task of traffic sign recognition, a transfer learning approach is employed. Transfer learning involves utilizing a pre-trained model from a large dataset for a different, but related, task. In this project, a pre-trained ResNet model from the torchvision package is repurposed for identifying traffic signs. In the transfer learning process using ResNet-18, the ultimate classification layer undergoes modification. This layer sequence begins with a Fully Connected Layer responsible for extracting final features and the subsequent Softmax layer undergoes modification.

## Dataset Composition
  The process of data collection for this project involves the meticulous assembly of a diverse dataset encompassing a wide array of traffic and road signs. The dataset is curated from various sources, ensuring a comprehensive representation of real-world scenarios and sign variations. As part of the data cleaning process, images with low resolution and those deemed irrelevant were systematically removed from the dataset. 
The elimination of low-resolution and unwanted images contributes to the overall data quality, enhancing the model's ability to learn and generalize effectively during the training phase.

The dataset is systematically classified into four distinct classes, each representing a major category of road signs. The classes include:
1. Traffic Light (Class 0)
2. Stop Sign (Class 1)
3. Speed Limit Sign (Class 2)
4. Crosswalk Sign (Class 3)

To optimize the learning outcomes and enhance the model's adaptability to diverse sources, various data transformations are applied to the collected samples. Few of the transformations include:<br/>
• Resizing: Ensuring uniform scaling of images to accommodate variations in source dimensions.<br/>
• Horizontal Flipping: Augmenting the dataset by introducing flipped versions of images, enhancing the model's robustness to different orientations.<br/>
• Normalization: Standardizing pixel values to a common scale, aiding in convergence during training. etc.,<br/>

## Results
The accuracy of our trained model stands impressively at an average of 97%. The accuracy of our trained model and prediction from a video stands at an average of 80%. This metric reflects the model's ability to correctly classify instances into their respective classes, indicating a high level of effectiveness.

Below is the sample result for the model learning and prediction analysis from images. The epoch-wise loss analysis provides valuable insights into the model's learning process. The gradual decrease in loss across successive epochs, signifies that the model is learning and optimizing effectively. The gradual decrease in loss over epochs indicates that the model is learning well, adapting to the intricacies of the training data.

Epoch 1, Loss: 1.010608493793206<br/>
Epoch 2, Loss: 0.5091489670706577<br/>
Epoch 3, Loss: 0.2372961545088252<br/>
Epoch 4, Loss: 0.12521728244228442<br/>
Epoch 5, Loss: 0.07695121708952013<br/>
Epoch 6, Loss: 0.05107131529385682<br/>
Epoch 7, Loss: 0.036296675836697954<br/>
Epoch 8, Loss: 0.025733748450875282<br/>
Epoch 9, Loss: 0.024913500555500876<br/>
Epoch 10, Loss: 0.01958332117265243<br/>

### Confusion Matrix Interpretation

<table>
    <tr>
        <th></th>
        <th>Actual Class 0</th>
        <th>Actual Class 1</th>
        <th>Actual Class 2</th>
        <th>Actual Class 3</th>
    </tr>
    <tr>
        <td>Predicted Class 0</td>
        <td>102</td>
        <td>0</td>
        <td>0</td>
        <td>7</td>
    </tr>
    <tr>
        <td>Predicted Class 1</td>
        <td>0</td>
        <td>148</td>
        <td>0</td>
        <td>1</td>
    </tr>
    <tr>
        <td>Predicted Class 2</td>
        <td>0</td>
        <td>1</td>
        <td>118</td>
        <td>0</td>
    </tr>
    <tr>
        <td>Predicted Class 3</td>
        <td>4</td>
        <td>0</td>
        <td>2</td>
        <td>106</td>
    </tr>
</table>

## Local Setup
These  are the instructions to set the project up and running on local machine for development and testing purposes.

### Prerequisites
Things needed to install the software.
 - Python 3.6 or higher
 - pip
   
### Installation
A step-by-step series of examples to get a development environment running.

#### Clone the repository to your local machine:
``` 
git clone https://github.com/krishnapuja/Traffic-Sign-Prediction.git
```
#### Navigate to the project directory:
```
cd Traffic-Sign-Prediction
```
#### Install the required dependencies:
```
pip install -r requirements.txt 
``` 

## Usage
How to use the project, including any scripts and commands.

To run the project that trains the model and predicts with images:
``` 
python ML_Project.py
``` 
To run the project that trains the model and predicts with video:
``` 
python ML_Project_Video.py
``` 

## Built With
 - PyTorch: The deep learning framework used
 - Torchvision: For utilities related to image loading and transformations
 - Scikit-learn: For accuracy score and confusion matrix
 - Pillow: For image manipulation
 - Matplotlib: For plotting
