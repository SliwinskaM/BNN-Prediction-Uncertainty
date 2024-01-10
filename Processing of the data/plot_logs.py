import matplotlib.pyplot as plt
import torch
import os

dataset = "KNEE"

print("#######################################################################################################################################################")

dropout_list = ["DROPOUT00", "DROPOUT02", "DROPOUT04", "DROPOUT06", "DROPOUT08"]
logs_list = ["log0.txt", "log1.txt", "log2.txt", "log3.txt", "log4.txt"]
dropout_dict = {"DROPOUT00": 0.0, "DROPOUT02": 0.2, "DROPOUT04":0.4, "DROPOUT06": 0.6, "DROPOUT08": 0.8}

epochs = [[], [], [], [], []]
train_loss = [[], [], [], [], []]
val_loss = [[], [], [], [], []]
val_dice = [[], [], [], [], []]

best_epochs = []

next_iter_debug = []


for dropout_idx in range(len(dropout_list)):
  dropout = dropout_list[dropout_idx]
  print("-----------" + dropout + "------------")
 ############### INDIVIDUAL ###############
  for log_file in logs_list:
    log_number = log_file[3:-4]

    epochs_local = [[]]
    train_loss_local = [[]]
    val_loss_local = [[]]
    val_dice_local = [[]]

    prev_epoch = -1
    iter = 0
    legend = ["Part 1"]

    with open(os.path.join(dataset, dropout, "MODELS", log_file)) as fp:
        lines = fp.readlines()
        for line in lines:
            words = line.split()
            epoch = int(words[1])
            if epoch < 2000:
              if epoch <= prev_epoch:
                iter += 1
                next_iter_debug.append(dropout + str(log_number))
                epochs_local.append([])
                train_loss_local.append([])
                val_loss_local.append([])
                val_dice_local.append([])
                legend.append("Part " + str(iter+1))

              epochs_local[iter].append(epoch)
              train_loss_local[iter].append(float(words[5]))
              val_loss_local[iter].append(float(words[9]))
              val_dice_local[iter].append(float(words[13]))

              prev_epoch = epoch

    # print(log_number, iter)

    fname = os.path.join(dataset, dropout, "MODELS/fold_" + str(log_number) + "_model_best.model")
    best_model = torch.load(fname, map_location=torch.device('cpu'))
    bestEpoch = best_model['epoch']
    bestDice = val_dice_local[-1][epochs_local[-1].index(bestEpoch)]
    bestTrain = train_loss_local[-1][epochs_local[-1].index(bestEpoch)]
    bestVal = val_loss_local[-1][epochs_local[-1].index(bestEpoch)]
    # print(bestEpoch, bestDice)
    legend.append("Best model")

    plt.figure()
    for i in range(iter+1):
      plt.plot(epochs_local[i], val_dice_local[i])
    plt.plot(bestEpoch, bestDice, marker="o", markersize=7, color="yellow")
    plt.title(dataset + " dataset, dropout p=" + str(dropout_dict[dropout]) + ", fold " + str(log_number) + "\n Dice coefficient for validation data")
    plt.legend(legend)
    plt.xlabel("Epoch")
    plt.ylabel("Dice coefficient")
    plt.savefig(os.path.join(dataset, dropout, "MODELS/dice_log" + str(log_number) + ".png"))
    plt.close()

    plt.figure()
    for i in range(iter+1):
      plt.plot(epochs_local[i], train_loss_local[i])
    plt.plot(bestEpoch, bestTrain, marker="o", markersize=7, color="yellow")
    plt.title(dataset + " dataset, dropout p=" + str(dropout_dict[dropout]) + ", fold " + str(log_number) + "\n Training loss")
    plt.legend(legend)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.savefig(os.path.join(dataset, dropout, "MODELS/train_loss_log" + str(log_number) + ".png"))
    plt.close()

    plt.figure()
    for i in range(iter+1):
      plt.plot(epochs_local[i], val_loss_local[i])
    plt.plot(bestEpoch, bestVal, marker="o", markersize=7, color="yellow")
    plt.title(dataset + " dataset, dropout p=" + str(dropout_dict[dropout]) + ", fold " + str(log_number) + "\n Validation loss")
    plt.legend(legend)
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.savefig(os.path.join(dataset, dropout, "MODELS/val_loss_log" + str(log_number) + ".png"))
    plt.close()









  ############### CUMULATIVE ###############
  epochs_for_logs = [[], [], [], [], []]
  train_loss_for_logs = [[], [], [], [], []]
  val_loss_for_logs = [[], [], [], [], []]
  val_dice_for_logs = [[], [], [], [], []]

  for log_file in logs_list:
    log_number = int(log_file[3:-4])
    print("LOG ", log_number)

    epochs_for_iters = [[]]
    train_loss_for_iters = [[]]
    val_loss_for_iters = [[]]
    val_dice_for_iters = [[]]


    prev_epoch = -1
    iter = 0

    with open(os.path.join(dataset, dropout, "MODELS", log_file)) as fp:
        lines = fp.readlines()
        for line in lines:
            words = line.split()
            epoch = int(words[1])
            if epoch <= prev_epoch:
              iter += 1
              epochs_for_iters.append([])
              train_loss_for_iters.append([])
              val_loss_for_iters.append([])
              val_dice_for_iters.append([])

            epochs_for_iters[iter].append(epoch)
            train_loss_for_iters[iter].append(float(words[5]))
            val_loss_for_iters[iter].append(float(words[9]))
            val_dice_for_iters[iter].append(float(words[13]))

            prev_epoch = epoch

    print("epochs_for_iters lengths ", [len(i) for i in epochs_for_iters])
    assert [len(i) for i in epochs_for_iters] == [len(i) for i in train_loss_for_iters] == [len(i) for i in val_loss_for_iters] == [len(i) for i in val_dice_for_iters]
    print("epochs_for_iters maxes ", [max(i) for i in epochs_for_iters])
    


    # average values through all iterations = for the whole log
    max_iter_epoch = max([max(iter_list) for iter_list in epochs_for_iters])
    print("max_iter_length ", max_iter_epoch)

    for i_epoch in range(max_iter_epoch):
        epochs_for_logs[log_number].append(i_epoch)
        
        # train loss
        sum_train_loss = 0
        count_train_loss = 0
        for iter in range(len(epochs_for_iters)):
          if i_epoch in epochs_for_iters[iter]: # len(train_loss_for_iters[iter]) > i:
             train_i = epochs_for_iters[iter].index(i_epoch)
             sum_train_loss += train_loss_for_iters[iter][train_i]
             count_train_loss += 1
        train_loss_for_logs[log_number].append(sum_train_loss / count_train_loss)

        # val loss
        sum_val_loss = 0
        count_val_loss = 0
        for iter in range(len(epochs_for_iters)):
          if i_epoch in epochs_for_iters[iter]: # len(val_loss_for_iters[iter]) > i:
             val_i = epochs_for_iters[iter].index(i_epoch)
             sum_val_loss += val_loss_for_iters[iter][val_i]
             count_val_loss += 1
        val_loss_for_logs[log_number].append(sum_val_loss / count_val_loss)

        # val dice
        sum_val_dice = 0
        count_val_dice = 0
        for iter in range(len(epochs_for_iters)):
          if i_epoch in epochs_for_iters[iter]: # len(val_dice_for_iters[iter]) > i:
             val_i = epochs_for_iters[iter].index(i_epoch)
             sum_val_dice += val_dice_for_iters[iter][val_i]
             count_val_dice += 1
        val_dice_for_logs[log_number].append(sum_val_dice / count_val_dice)


    fname = os.path.join(dataset, dropout, "MODELS/fold_" + str(log_number) + "_model_best.model")
    best_model = torch.load(fname, map_location=torch.device('cpu'))
    best_epochs.append(best_model['epoch'])
    # best_dice_for_log = val_dice_for_logs[-1][epochs[-1].index(best_epoch_for_log)]
    # best_train_for_log = train_loss_for_logs[-1][epochs[-1].index(best_epoch_for_log)]
    # best_val_for_log = val_loss_for_logs[-1][epochs[-1].index(best_epoch_for_log)]
    # best_dices[best_epoch_for_log] = best_dice_for_log
    # best_trains[best_epoch_for_log] = best_train_for_log
    # best_vals[best_epoch_for_log] = best_val_for_log



  print("-----")
  print("epochs_for_logs lengths ", [len(i) for i in epochs_for_logs])
  print("lengths equal: ", [len(i) for i in epochs_for_logs] == [len(i) for i in train_loss_for_logs] == [len(i) for i in val_loss_for_logs] == [len(i) for i in val_dice_for_logs])
  print("epochs_for_logs maxes ", [max(i) for i in epochs_for_logs])
  


  # average values for every dropout rate
  max_log_epoch = max([max(log_list) for log_list in epochs_for_logs])
  print("max_log_epoch ", max_log_epoch)

  for l_epoch in range(max_log_epoch):
    epochs[dropout_idx].append(l_epoch)
    
    # train loss
    sum_train_loss = 0
    count_train_loss = 0
    for log in range(len(epochs_for_logs)):
      if l_epoch in epochs_for_logs[log]: # len(train_loss_for_logs[log]) > i:
          j_train = epochs_for_logs[log].index(l_epoch)
          sum_train_loss += train_loss_for_logs[log][j_train]
          count_train_loss += 1
    train_loss[dropout_idx].append(sum_train_loss / count_train_loss)

    # val loss
    sum_val_loss = 0
    count_val_loss = 0
    for log in range(len(epochs_for_logs)):
      if l_epoch in epochs_for_logs[log]: # len(val_loss_for_logs[log]) > i:
          j_val = epochs_for_logs[log].index(l_epoch)
          sum_val_loss += val_loss_for_logs[log][j_val]
          count_val_loss += 1
    val_loss[dropout_idx].append(sum_val_loss / count_val_loss)

    # val dice
    sum_val_dice = 0
    count_val_dice = 0
    for log in range(len(epochs_for_logs)):
      if l_epoch in epochs_for_logs[log]: # len(val_dice_for_logs[log]) > i:
          j_val = epochs_for_logs[log].index(l_epoch)
          sum_val_dice += val_dice_for_logs[log][j_val]
          count_val_dice += 1
    val_dice[dropout_idx].append(sum_val_dice / count_val_dice)


# furthest_best_dice = max(best_dices.keys())
# furthest_best_train = max(best_trains.keys())
# furthest_best_val = max(best_vals.keys())


print("epochs lengths ", [len(i) for i in epochs])
assert [len(i) for i in epochs] == [len(i) for i in train_loss] == [len(i) for i in val_loss] == [len(i) for i in val_dice]
print("epochs maxes ", [max(i) for i in epochs])
print("!!! Furthest best model's epoch: ", max(best_epochs), " !!!")


plt.figure()
for i in range(5):
  plt.plot(epochs[i], val_dice[i], label="p=" + str(dropout_dict[dropout_list[i]]))
plt.title("Dice coefficient for validation data")
plt.xlabel("Epoch")
plt.ylabel("Dice coefficient")
plt.legend()
plt.savefig(os.path.join(dataset, "plots/dice.png"))
plt.close()

plt.figure()
for i in range(5):
  plt.plot(epochs[i], train_loss[i], label="p=" + str(dropout_dict[dropout_list[i]]))
plt.title("Training loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.savefig(os.path.join(dataset, "plots/train_loss.png"))
plt.close()

plt.figure()
for i in range(5):
  plt.plot(epochs[i], val_loss[i], label="p=" + str(dropout_dict[dropout_list[i]]))
plt.title("Validation loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Validation loss")
plt.savefig(os.path.join(dataset, "plots/val_loss.png"))
plt.close()

plt.figure()
plt.hist(best_epochs)
plt.title("Epochs where best models were achieved")
plt.xlabel("Epoch")
plt.ylabel("Number of models")
plt.savefig(os.path.join(dataset,"plots/best_epochs.png"))
plt.close()





# print(next_iter_debug)