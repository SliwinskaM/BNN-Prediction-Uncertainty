import os
import datetime

threshold = datetime.datetime(2023, 11, 23, 0, 0, 0)
wrong_time = []

for dataset in ["KNEE", "SKB"]:
    for dropout in ["DROPOUT00", "DROPOUT02", "DROPOUT04", "DROPOUT06", "DROPOUT08"]:
        if dropout == "DROPOUT00":
            preds = sorted(os.listdir(os.path.join(dataset, dropout, "predsTs")))
            # print(preds)
            for pred in preds:
                # print(pred)
                # mod_time =  os.path.getmtime(pred)
                pred_path = os.path.join(dataset, dropout, "predsTs", pred)
                stat_info = os.stat(pred_path)
                mod_time = datetime.datetime.fromtimestamp(stat_info.st_mtime)
                # print(dataset, dropout, pred, mod_time)
                if mod_time < threshold: # and (dataset=="KNEE" and dropout=="DROPOUT00"):
                    wrong_time.append([pred_path, mod_time])
                    
        else:
          for iter in range(10):
              preds = sorted(os.listdir(os.path.join(dataset, dropout, "predsTs_" + str(iter))))
              # print(preds)
              for pred in preds:
                  # print(pred)
                  # mod_time =  os.path.getmtime(pred)
                  pred_path = os.path.join(dataset, dropout, "predsTs_" + str(iter), pred)
                  stat_info = os.stat(pred_path)
                  mod_time = datetime.datetime.fromtimestamp(stat_info.st_mtime)
                  # print(os.path.join(dataset, dropout, "predsTs_" + str(iter), pred), mod_time)
                  if mod_time < threshold: # and ((dataset=="LUNG" and dropout=="DROPOUT04") or \
                  #                             (dataset=="LUNG" and dropout=="DROPOUT06") or \
                  #                             (dataset=="LUNG" and dropout=="DROPOUT08") or \
                  #                             (dataset=="KNEE" and dropout=="DROPOUT02") or \
                  #                             (dataset=="KNEE" and dropout=="DROPOUT04") or \
                  #                             (dataset=="KNEE" and dropout=="DROPOUT06") or \
                  #                             (dataset=="SKB" and dropout=="DROPOUT08")):
                      wrong_time.append([pred_path, mod_time])

print("WRONG:")
for pair in wrong_time:
    print(pair)