import dataloader
from GA import GA2
import time

start = time.process_time()

df, data, label, symbol, col_name = dataloader.dataloader()

human = GA2.GA().initialization()
avg, X_C, rawData_std = GA2.GA().seperatezerotoone(label, data, human)

rank, final_p_value = GA2.GA().rank(avg, X_C)

new_human = GA2.GA().evolve(human, rawData_std, X_C, data)
csvfile = GA2.GA().savescv(new_human, df, label, symbol, col_name)

print("process time = ", time.process_time() - start, 'sec')
