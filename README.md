# F20DL
F20DL


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
Random shuffle
weka.filters.supervised.instance.Resample - only works on nominal
weka.filters.unsupervised.instance.Randomize

Remove Instances
Removing 70% of dataset.
20% for training.
10% for testing.


## Refrences
https://journal.r-project.org/archive/2009/RJ-2009-016/RJ-2009-016.pdf

http://weka.sourceforge.net/doc.stable/weka/filters/unsupervised/instance/package-summary.html

http://weka.sourceforge.net/doc.stable/weka/filters/unsupervised/instance/Resample.html


# Step 4

Reduced data set into two sizes - 10% and 20%.

100% results in 58.7% (training set) - no filters

10% resample results in 56% (training set) - Filter Descritze

20% resample results in 57.5% (training set) - Filter Descritze

20% resample results in 55.6% (training set) - no filters

20% resample results in 55.5% (training set) - no filters

20% resample results in 57.5% (training set) - Filter Standardize

20% resample results in 57.5% (training set) - Filter Normalize

20% resample results in 30.6% (training set = 10% resample) - Filter Descritze

Turn these results into a table and save models
