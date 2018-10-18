# F20DL - Data Mining CW1 Report

F20DL_2018-2019

Data Mining and Machine Learning

Taught by Diana Bental and Ekaterina Komendantskaya


# Team
Mobeen Aftab

Jonathan Mendoza

Owen Welch


# File Structure
- Python
    - Holds python scripts
    - convert.py
        - Convert file into csv format for training and testing, step 1 -3
    - main.py
        - Main python script performing steps 3 - 6
    - deeper_analysis.py
        - Bring together all the selectKBest attributes and run on fer2018
- R
    - script.R
        - Using RWeka to convert csv file into arff format for weka compatibility, step 1
- images
    - Images of results, testing and setting options for each step.

# steps 1 - 3

## Step 1
Using Excel:
Deleted row headers in every .csv file.
Selected second column containing attribute values.
Used the "Text to Columns" command.
Delimited > space > General > Finish.

## Step 2
Used a script written in R to automatically generate, and format .csv files into .arff file.
Manually added the class attribute to each arff file.
R file has comments explaing code.
Libraries used: farff.
Has a method called 'writeARFF', We used this over the 'RWeka.write.arff()' because more optional parameter controlls and uses the internal 'write.table' method making is much faster than RWeka's write.arff (also removes heap size limit).

## Step 3

### Weka GUI
Random shuffle
weka.filters.supervised.instance.Resample - only works on nominal
weka.filters.unsupervised.instance.Randomize

#### Weka File setup
Remove Instances
Removing 70% of dataset.
20% for training.
10% for testing.

### scikit-learn
Using [scikit-learn](http://scikit-learn.org/stable/) we wrote a [python script](https://github.com/MobeenAftab/F20DL/blob/master/Python/convert.py) to convert the file into a proper csv format which can then be used with scikit-learn.

# Step 4 - 6

Our [main python script](https://github.com/MobeenAftab/F20DL/blob/master/Python/main.py) contains the code to perform steps 4 - 6.

To configure what dataset to use you must manualy set the `input_file_path ` variable and select one of the emotion filepaths variables.

The `main()` function will call the function to train and test KNN. Comment in or out the functions you wish to call in here.

Every method is commented and explains its purpose.

We also used weka along with our python script to compare both results and found that througout testing both Weka and scikit-learn produced very similar results.

See the image folder for detailed results of each section.

## Refrences

[Our Google Drive Folder](https://drive.google.com/drive/folders/1SyGKWrumyfmoDq8m4GqKGBNMXvbOoe87?usp=sharing)

[Mwagha, S., Muthoni, M. and Ochieg, P. (2014). Comparison of Nearest Neighbor (ibk), Regression by Discretization and Isotonic Regression Classification Algorithms for Precipitation Classes Prediction. [online] Ir.cut.ac.za.](http://ir.cut.ac.za/bitstream/handle/11462/723/Comparison%20of%20Nearest%20Neighbor%20%28ibk%29%2C%20Regression%20by%20Discretization%20and%20Isotonic%20Regression%20Classification%20Algorithms%20for%20Precipitation%20Classes%20Prediction.pdf?sequence=1&isAllowed=y)

[Scikit-learn.org. (2018). sklearn.feature_selection.f_classif — scikit-learn 0.20.0 documentation. [online]](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif)

[Williams, G. (2009). Rattle: A Data Mining GUI for R. [online] Journal.r-project.org.](https://journal.r-project.org/archive/2009/RJ-2009-016/RJ-2009-016.pdf)

[Weka.sourceforge.net. (2018). weka.filters.unsupervised.instance. [online]](http://weka.sourceforge.net/doc.stable/weka/filters/unsupervised/instance/package-summary.html)

[Rayward-Smith, V. (2007). Statistics to measure correlation for data mining applications. Computational Statistics & Data Analysis, [online] 51(8), pp.3968-3982.](https://www.sciencedirect.com/science/article/pii/S0167947306001897)


https://journal.r-project.org/archive/2009/RJ-2009-016/RJ-2009-016.pdf

http://weka.sourceforge.net/doc.stable/weka/filters/unsupervised/instance/package-summary.html

http://weka.sourceforge.net/doc.stable/weka/filters/unsupervised/instance/Resample.html

