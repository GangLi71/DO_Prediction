1. Environment configuration: The environment and installation packages required for project operation can be found at requirements.txt.

2. Data description: The data folder contains three datasets (ori_data.csv, wq_data_0.7.csv, wq_data_0.8.csv). The ori_data.csv represents all of the datasets used for the research, the wq_data_0.7.csv represents the 70% of the datasets used for the data generation, and the wq_data_0.8.csv represents the 80% of the datasets used for the data generation. Users can alternate their own datasets for the specific research.

3. Data generation module: Use the main_timegan.py to achieve data generation, and then the generated data is saved in the generated_data.npy.

4. Visualization module: Use the visualization module to perform PCA and t-SNE analysis between qriginal data and synthetic data.

5. Prediction module: Use TimeGAN-LSTM to add different proportions of the synthetic data for LSTM model training and prediction. The prediction results during training period and testing period are saved in 0T_train.csv and 0T_test.csv, repsectively.
