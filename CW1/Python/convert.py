file = 'fer2018surprise.csv'
file_path = '../data/original/'+file
new_file_path = '../data/converted/'+file

num = 0

with open(file_path, 'r') as fp:
    with open(new_file_path, 'w') as fp2:
        num = num + 1
        line = fp.readline()
        while line:
            lineWithCommas = line.replace(' ', ',')
            fp2.write(lineWithCommas)
            line = fp.readline()
            num = num+1
            if num % 100 == 0:
                print("Working on line", num)


print("DONE!")