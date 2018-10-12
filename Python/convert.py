num = 0
with open('../data/fer2018.csv', 'r') as fp:
    with open('../data/fer2018-converted.csv', 'w') as fp2:
        num = num + 1
        line = fp.readline()
        while line:
            lineWithCommas = line.replace(' ', ',')
            fp2.write(lineWithCommas)
            line = fp.readline()
            num = num+1
            if num % 10 == 0:
                print("Working on line", num)


print("DONE!")