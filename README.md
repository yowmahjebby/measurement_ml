# Body Measurement Prediction Model

This project uses machine learning to predict **bust_circumference**, **waist_circumference**, and **hip_circumference** based on features like gender, age, height, weight, and other engineered ratios.

Make sure you have **Python 3.7+** installed.

Then install all necessary dependencies using:

```bash
pip install -r requirements.txt

#######################################################################################
#######################################################################################
Measurement ML (Project Structure)
├── train.csv
├── test.csv
├── model.pkl               # Trained model file
├── predict.py              # Script to make predictions
├── requirements.txt
└── README.md
└── measurements_nb.ipynb   # Training script
#######################################################################################
#######################################################################################

For the following files, each CSV files should only have the following columns:

train.csv — contains gender, age, height, weight, bust_circumference, waist_circumference, hip_circumference measurements.

test.csv — contains only gender, age, height, weight


To run the prediction make sure you run this on the root of the project folder

cd path/to/your/project

and then

```bash
python predict.py