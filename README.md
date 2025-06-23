This project focuses on cleaning and preprocessing the Titanic dataset in preparation for machine learning. The objective is to demonstrate how raw data can be refined by handling missing values, encoding categorical features, standardizing numerical columns, detecting outliers, and saving the cleaned dataset. The entire process is performed using Python in Visual Studio Code (VS Code), and uses essential data science libraries such as Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn.

The dataset used is the Titanic dataset, which is available publicly on Kaggle. The train.csv file from the competition was used, renamed to titanic.csv, and placed in the project folder alongside the Python script and this README file.

The project folder includes three key files: the raw titanic.csv dataset, the Python script named titanic_cleaning.py, and this README.md file. Before starting, it's important to install the required libraries if they're not already available. You can do this by running pip install pandas numpy matplotlib seaborn scikit-learn in the terminal inside VS Code.

The main script performs several key steps. First, it loads the dataset using Pandas and prints basic information such as column types and missing values. Next, it handles missing values by filling the Age column with its median, the Embarked column with its mode, and dropping the Cabin column due to excessive missing data. After this, it converts the Sex column to numerical values using label encoding, and applies one-hot encoding to the Embarked column to create separate binary columns.

For numerical preprocessing, the script uses Scikit-learn’s StandardScaler to standardize the Age and Fare columns. Outliers in these columns are then visualized using Seaborn boxplots. To remove them, the script applies the Interquartile Range (IQR) method and filters out extreme values. After all preprocessing steps are complete, the cleaned dataset is saved as cleaned_titanic.csv in the same project folder.

There are a couple of common errors you may encounter. One is a pandas.errors.EmptyDataError, which occurs if the titanic.csv file is empty or invalid. This can be fixed by re-downloading the correct CSV file from Kaggle. Another frequent error is NameError: name 'plt' is not defined, which happens when Matplotlib is used without importing it. To fix this, ensure you include import matplotlib.pyplot as plt at the beginning of your script.

To run the script, open the terminal in VS Code and type python titanic_cleaning.py. The script will execute the full preprocessing pipeline, show a boxplot for visual inspection of outliers, and create a new file named cleaned_titanic.csv, which contains the final clean data ready for machine learning tasks.

This project is ideal for students or beginners learning data science who want to understand the fundamentals of data cleaning using a real-world dataset. It lays the foundation for further steps such as model building, evaluation, and deployment.

