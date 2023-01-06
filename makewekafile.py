import dataloader

df, data, label, symbol, col_name = dataloader.dataloader()
name = []
f = open("newGA_parkinson_weka.arff", 'w')
f.write("@relation parkinson" + '\n')
for a in range(len(symbol)):
    if symbol[a] in name:
        f.write("@attribute " + symbol[a] + str(a) + " numeric" + '\n')
    else:
        f.write("@attribute " + symbol[a] + " numeric" + '\n')
    name.append(symbol[a])

f.write("@attribute 'class' {'1', '0'}" + '\n')
f.write("@data" + '\n')

for i in range(73):
    for j in range(17581):
        if j == 17580:
            f.write(str(int(df[j][i])))
        else:
            f.write(str(df[j][i]) + ", ")
    f.write('\n')

f.close()
print("Done")
