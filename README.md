## Traffic4cast 2021 – Temporal and Spatial Few-Shot Transfer Learning in Traffic Map Movie Forecasting
Study high-resolution 8-channel traffic movies of entire cities.  
Overcome the temporal domain shift pre/post COVID-19  
For more details about the challenge and how you can download the data, see [here](https://github.com/iarai/NeurIPS2021-traffic4cast)

# explain the problem, why we used unet. evry tings? handing the huge data sets?

## Project structure:
- models
    - unet.py
- utils
    - dataset.py
    - util.py
- inference.py
- train.py

## Techniques Utilized
#### Random Sampling of Data:  
Due to the large size of our dataset and computational power constraints, we implemented random sampling. This involved selecting a representative subset of our data for training and testing. By doing so, we were able to strike a balance between utilizing sufficient data for accurate model training and not overburdening our resources.

#### Learning Rate Scheduler:  
We employed a learning rate scheduler during training. Learning rate scheduling involves adjusting the learning rate during training to improve convergence and optimize the training process. By gradually reducing the learning rate over time, we allowed the model to fine-tune its parameters more effectively as it approached convergence, leading to potentially better predictions.

#### Mixed Precision Training:  
Mixed precision training involves using a combination of single-precision and half-precision (float16) data types during model training. This technique is particularly beneficial for optimizing memory usage and training speed on modern GPUs that support mixed precision. By utilizing float16 for activations and gradients while keeping certain parameters in float32, we accelerated our training process without sacrificing model accuracy.

#### Gradient Accumulation:  
To mitigate the limitations posed by memory constraints, we implemented gradient accumulation. Instead of updating the model's weights after every batch, gradient accumulation allowed us to accumulate gradients over multiple mini-batches before performing a single update. This enabled us to effectively use larger batch sizes without memory overflow issues, leading to more stable and efficient training.

## Prediction vs Ground Truth
![111](https://github.com/KianoushAmirpour/Short_Term_Traffic_Forcast/assets/112323618/fb2d7c72-d353-41d0-93fb-8bd65da09862) ![222](https://github.com/KianoushAmirpour/Short_Term_Traffic_Forcast/assets/112323618/737a8f67-4465-4377-a566-204ec2002a65)

## Available Command Line Arguments
`--cities`:  
Description: List of cities to train and validate the model on.  
Use Case: Specify the cities you want to include in your training and validation process. Choose from 'ANTWERP', 'BANGKOK', 'BARCELONA', and 'MOSCOW' to customize your training dataset.    
`--train_year` and `--val_year`:  
Description: List of years for training and validation data, respectively.  
Use Case: Select the years of data you want to use for training and validation. Choose from 2019 and 2020 to tailor your dataset to specific time frames.  
`--model`:  
Description: Model architecture to use for predictions.  
Use Case: Choose the model architecture for your traffic prediction. The available option is "UNET," which you can further modify or expand as needed.  
`--scheduler`, `--learning_rate`, `--batch_size`, `--num_workers`, `--num_epochs`, `--L1_regularization`, `--wd`:    
Description: Hyperparameters for training and optimization.  
Use Case: Adjust these hyperparameters to fine-tune your training process. Experiment with learning rates, batch sizes, regularization options, weight decay, and other parameters to achieve optimal results.  
`--num_file_train`, `--accumulation_step`, `--use_mask`:   
Description: Additional training-related parameters.  
Use Case: Control the number of files used for training, the gradient accumulation step size, and whether to use masks in your training data.  
`--device`: 
Description: Device for training ('cuda:0' for GPU or 'cpu' for CPU).  
Use Case: Select the device on which to train your model. This argument allows you to leverage available hardware resources effectively.  
`--seed`:  
Description: Seed for random number generation.  
Use Case: Set a specific seed to ensure reproducibility across different runs of the code.  


