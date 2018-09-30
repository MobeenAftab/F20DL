# F20DL
F20DL


# fer2018

## Steps Taken
Using Excel:
Deleted row headers in every .csv file.
Selected second column containing attribute values.
Used the "Text to Columns" command.
Delimited > space > General > Finish.

Used a script written in R to automatically generate, and format .csv files into .arff file.
Manually added the class attribute to each arff file.
R file has comments explaing code.
Libraries used: farff.
Has a method called 'writeARFF', We used this over the 'RWeka.write.arff()' because more optional parameter controlls and uses the internal 'write.table' method making is much faster than RWeka's write.arff (also removes heap size limit).