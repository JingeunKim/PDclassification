import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
human_number = 100
gene_number = 50
generation_num = 100


class GA():
    def __init__(self):
        self.human_number = human_number
        self.gene_number = gene_number
        self.generation_num = generation_num
        self.row = 73
        self.col = 17580

    def initialization(self):
        human = np.zeros((human_number, self.col))
        for j in range(human_number):
            createNum = self.createNumber()
            for i in range(gene_number):
                human[j][createNum[i]] = 1
        return human

    def createNumber(self):
        nb = []
        rnum = random.randint(0, self.col - 1)
        for i in range(gene_number):
            while rnum in nb:
                rnum = random.randint(0, self.col - 1)
            nb.append(rnum)
        return nb

    def score(self, human, rawData_std, rawData):
        avg = np.zeros(human.shape[0])
        for i in range(human.shape[0]):
            score_sum = 0
            new_df = human[i, :]
            indx = np.where(new_df == 1)
            indx = indx[0]
            for x in range(self.gene_number - 1):
                idx = indx[x]
                for a in range(x + 1, self.gene_number):
                    idx2 = indx[a]
                    x_std = rawData_std[idx]
                    y_std = rawData_std[idx2]

                    dataset = pd.DataFrame(rawData)
                    x = dataset.iloc[:, idx]
                    x = np.array(x)

                    y = dataset.iloc[:, idx2]
                    y = np.array(y)
                    score_cov = np.cov(x, y)[0, 1]
                    p_x_y = abs(score_cov / (x_std * y_std))
                    score_sum += p_x_y
                avg[i] = score_sum / (self.gene_number * (self.gene_number - 1))
        return avg

    def seperatezerotoone(self, label, rawData, human):
        oneCount = 0
        zeroCount = 0
        label = label.to_numpy()
        rawData = rawData.to_numpy()
        for i in range(self.row):
            if label[i] == 1:
                oneCount += 1

            elif label[i] == 0:
                zeroCount += 1
        new_rawData_zero = np.zeros((zeroCount, self.col))
        new_rawData_one = np.zeros((oneCount, self.col))
        for a in range(oneCount):
            for i in range(self.row):
                if label[i] == 1:
                    for j in range(self.col - 1):
                        new_rawData_one[a][j] = rawData[i][j]

        for a in range(zeroCount):
            for i in range(self.row):
                if label[i] == 0:
                    for j in range(self.col - 1):
                        new_rawData_zero[a][j] = rawData[i][j]
        X_C = self.cal(new_rawData_zero, new_rawData_one)
        rawData_mean = rawData.mean(axis=0)
        rawData_std = rawData.std(axis=0)
        avg = self.score(human, rawData_std, rawData)

        return avg, X_C, rawData_std

    def cal(self, new_rawData_zero, new_rawData_one):

        new_rawData_zero_mean = new_rawData_zero.mean(axis=0).mean()
        new_rawData_one_mean = new_rawData_one.mean(axis=0).mean()
        new_rawData_zero_std = new_rawData_zero.std(axis=0).mean()
        new_rawData_one_std = new_rawData_one.std(axis=0).mean()

        X_C = abs((new_rawData_zero_mean - new_rawData_one_mean) / (new_rawData_zero_std + new_rawData_one_std))

        X_C = X_C ** -1
        return X_C

    def rank(self, avg, X_C):
        final_p_value = np.zeros(len(avg))
        for i in range(len(avg)):
            # final_p_value[i] = avg[i] * 2 + X_C
            final_p_value[i] = (X_C*self.gene_number)/math.sqrt(self.gene_number+self.gene_number*(self.gene_number-1)*avg[i])
        rank = final_p_value.argsort()[::-1]
        return rank, final_p_value

    def evolve(self, human, rawData_std, X_C, rawData):
        print("evolve START1")
        rawData = rawData.to_numpy()
        human_range = self.human_number
        new_human = human

        threshold_GA = []

        for p in range(self.generation_num - 1):
            print(p + 1, "세대")
            for j in range(int(human_number / 2)):
                rand_num = random.randint(0, self.col - 1)
                if j == human_number / 4:
                    mom = new_human[0 + j]
                    dad = new_human[0]
                else:
                    # if j
                    mom = new_human[2 * j]
                    dad = new_human[2 * j + 1]

                baby1 = np.concatenate((mom[rand_num:], dad[:rand_num]), axis=None)

                baby2 = np.concatenate((mom[:rand_num], dad[rand_num:]), axis=None)

                baby1 = self.oneDetector(baby1)
                baby1 = baby1.reshape(1, -1)
                new_human = np.concatenate((new_human, baby1), axis=0)

                baby2 = self.oneDetector(baby2)
                baby2 = baby2.reshape(1, -1)
                new_human = np.concatenate((new_human, baby2), axis=0)

                mutate_pro = 0.1
                mutate_rand = random.random()
                if mutate_pro > mutate_rand:
                    baby1_one_index = np.where(baby1 == 1.0)
                    baby1_one_index = list(baby1_one_index[1])
                    select_mutate_point = random.randint(0, gene_number - 2)
                    baby1 = baby1.flatten()

                    b = baby1_one_index[select_mutate_point]
                    baby1[b] = 0

                    baby1_zero_index = np.where(baby1 == 0.0)
                    baby1_zero_index = list(baby1_zero_index[0])
                    select_mutate_point = random.randint(0, gene_number - 2)
                    a = baby1_zero_index[select_mutate_point]
                    baby1[a] = 1
            avg = self.score(new_human, rawData_std, rawData)
            rank, final_p_value = self.rank(avg, X_C)
            new_human = new_human[rank]
            new_human = new_human[:human_range]
            print("최고 높은 p value = ", final_p_value.max())
            threshold_GA.append(final_p_value.max())
        self.drawGA(threshold_GA)
        return new_human

    def oneDetector(self, baby):
        oneDetector = np.where(baby == 1)
        oneDetector = oneDetector[0]
        if len(oneDetector) < gene_number:
            for a in range(gene_number - len(oneDetector)):
                zerotoone = np.where(baby == 0)
                zerotoone = zerotoone[0]
                rand_num = random.randint(0, gene_number - 2)
                b = zerotoone[rand_num]
                baby[b] = 1
        elif len(oneDetector) > gene_number:
            for a in range(len(oneDetector) - gene_number):
                onetozero = np.where(baby == 1)
                onetozero = onetozero[0]
                rand_num = random.randint(0, gene_number - 2)
                b = onetozero[rand_num]
                baby[b] = 0
        elif len(oneDetector) == gene_number:
            return baby
        return baby

    def savescv(self, new_human, df, label, symbol, col_name):
        f = open("newGA2_parkinson_" + str(human_number) + "_" + str(gene_number) + "_" + str(generation_num) + ".csv",
                 'w')
        micro_array = []
        final_human = new_human[0]
        # print(col_name)
        for a in range(self.col):
            if final_human[a] == 1:
                data = df.iloc[:, a]
                # print(col_name[a])
                f.write(str(col_name[a]) + '\t')
                micro_array.append(df.columns[a])
                for n in range(self.row):
                    f.write(str(data[n]))
                    f.write('\t')

                f.write('\r\n')

        class_input = df.iloc[:, self.col]
        f.write("class" + '\t')
        for n in range(self.row):
            f.write(str(label[n]) + '\t')
        f.close()

        f = open("genetic_name2_" + str(human_number) + "_" + str(gene_number) + "_" + str(generation_num) + ".csv", 'w')
        f.write("SYMBOL" + '\t')
        for a in range(self.col):
            if final_human[a] == 1:
                f.write(str(symbol[n]) + '\t')
        f.close()

    def drawGA(self, value):
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.plot(value)
        plt.show()
