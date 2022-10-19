import dataloader
from GA import GA
import time

start = time.process_time()

df, data, label, symbol, col_name = dataloader.dataloader()

human = GA.GA().initialization()
avg, X_C, rawData_std = GA.GA().seperatezerotoone(label, data, human)

rank, final_p_value = GA.GA().rank(avg, X_C)

new_human = GA.GA().evolve(human, rawData_std, X_C, data)
csvfile = GA.GA().savescv(new_human, df, label, symbol, col_name)

print("process time = ", time.process_time() - start, 'sec')
