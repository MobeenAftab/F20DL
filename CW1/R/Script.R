# Import libraries
library("farff")
library("RWeka")
library("RWekajars")
#search()

# Set heap size to 6GB
options(java.parameters = "-Xmx6g")

# Must call rJava after heap size is set for java
library(rJava)

# Uncomment to check current heap size or use this function to set a new heap size
# memory.limit()

# Check if packages installed
#installed.packages()

#' Title
#' Call to generate files
#' @return null
#' @export arff file format
#'
#' @examples run function, select file, select/create output file
generateFile = function() {
    print("####### Process Start #######")

    # Create data table from csv data
    fer2018 <- read.csv(file.choose(), header = FALSE, row.names = NULL, encoding = "UTF-8", sep = ",", dec = ".", quote = "\"", comment.char = "")

    # Write fer2018 table data as arff file with generated attrubte headers
    writeARFF(fer2018, file.choose(), overwrite = FALSE, chunk.size = 1e+06, relation = deparse(substitute(x)))

    print("####### Process End #######")
}


