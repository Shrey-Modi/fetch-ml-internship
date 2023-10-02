# Fetch Machine Learning Internship Take Home Excercise

Welcome to the Fetch Machine Learning Internship project repository! This repository contains code and resources for a machine learning model developed during the internship.

## Getting Started

You can run the project either using Docker or by installing Python locally.

### Using Docker

1. Build the Docker image or pull it from Docker Hub([image link](https://hub.docker.com/r/shreymodi/fetch-ml)):

   ```bash
   # Build the Docker image (replace '.' with the path to your project directory)
   docker build . -t shreymodi/fetch-ml

   # OR pull the Docker image from Docker Hub
   docker pull shreymodi/fetch-ml
   ```

2. Run the Docker container:

   ```bash
   docker run -p 8501:8501 shreymodi/fetch-ml
   ```

3. Open your web browser and navigate to [http://0.0.0.0:8501](http://0.0.0.0:8501) to use the model.

### Using Python Locally

1. Clone this repository and navigate to the project directory:

   ```bash
   git clone https://github.com/Shrey-Modi/fetch-ml-internship.git .
   ```

2. Install the required Python packages:

   ```bash
   pip3 install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Open your web browser and navigate to [http://0.0.0.0:8501](http://0.0.0.0:8501) to use the model.

## Model Testing

Model testing and analysis are performed in the `testing.ipynb` Jupyter Notebook. This notebook includes:

- Data analysis and preprocessing steps.
- Testing different machine learning models.
- Explanation of the PyTorch linear model used in the final implementation.

## PyTorch Linear Model

The final model implemented for this project is a PyTorch linear model. The model's weights were calculated using the error normal method. This method optimizes the model's parameters by minimizing the mean squared error (MSE) between the model's predictions and the actual target values. The linear model aims to find the optimal weights for the given input features to make accurate predictions.

Feel free to explore the code, data, and analysis in this repository.
