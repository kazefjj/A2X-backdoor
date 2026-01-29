import csv
import os
def log_results_to_csv(accuracy, asr,dir):
        # 如果文件不存在，写入表头
    file_exists = os.path.exists(dir)

    with open(dir, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([ 'accuracy', 'ASR'])
        writer.writerow([ accuracy, asr])
def log_IBD_to_csv(TPR, FPR,AUC,dir):
        # 如果文件不存在，写入表头
    file_exists = os.path.exists(dir)

    with open(dir, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([ 'TPR', 'FPR','AUC'])
        writer.writerow([ TPR, FPR,AUC])