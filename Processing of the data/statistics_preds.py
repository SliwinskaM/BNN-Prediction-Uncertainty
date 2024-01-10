import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



# memory saving switches
method1 = False
method2 = False
method3 = True

dice_table_string = ""

q95_table_string = ""
q90_table_string = ""
q75_table_string = ""
q50_table_string = ""


for dataset in ["LUNG", "KNEE", "SKB"]:
  print("**************************************************", dataset, "******************************************************")
  test_labels = sorted(os.listdir(dataset + '/labelsTs')) # ['CT_0019.nii.gz', 'CT_0020.nii.gz', 'CT_0021.nii.gz', 'CT_0021.nii.gz', 'CT_0023.nii.gz'] #  # 

  ################################# MONTE CARLO DROPOUT #################################
  print("########### MONTE CARLO DROPOUT ###########")
  # method 1: difference between every prediction image and grund truth (one value for each image)
  diff_list_bayesian = []
  # method 1: difference between every prediction image and grund truth (one value for each prediction)
  dropout_diffs = {"DROPOUT02": [], "DROPOUT04": [], "DROPOUT06": [], "DROPOUT08": []}
  # method 1: difference between mean prediction image for the whole dropout and grund truth (one value for each *mean* prediction from all iterations for one dropout)
  dropout_means = {"DROPOUT02": [], "DROPOUT04": [], "DROPOUT06": [], "DROPOUT08": []}


  # method 2: dice between mean prediction image for the whole dropout and grund truth (one value for each *mean* prediction from all iterations for one dropout)
  dropout_dices = {"Traditional": [], "DROPOUT02": [], "DROPOUT04": [], "DROPOUT06": [], "DROPOUT08": []}


  # method3: quantiles of standard deviation between mean prediction image for the whole dropout and grund truth (one value for each *mean* prediction from all iterations for one dropout)
  std_all_pixels = [] # zobaczymy, czy program to przeżyje xd
  dropout_q95 = {"DROPOUT02": [], "DROPOUT04": [], "DROPOUT06": [], "DROPOUT08": []}
  dropout_q90 = {"DROPOUT02": [], "DROPOUT04": [], "DROPOUT06": [], "DROPOUT08": []}
  dropout_q75 = {"DROPOUT02": [], "DROPOUT04": [], "DROPOUT06": [], "DROPOUT08": []}
  dropout_q50 = {"DROPOUT02": [], "DROPOUT04": [], "DROPOUT06": [], "DROPOUT08": []}





  number_of_files = 0

  for label_name in test_labels:
      if True: # number_of_files < 10:
        number_of_files += 1
        # print("-------", label_name, "-------")
        # load label
        truth_path = os.path.join(dataset, 'labelsTs', label_name)
        truth_nifti = nib.load(truth_path)
        truth_matrix = truth_nifti.get_fdata() # 3D numpy matrix with an image
        affinity = truth_nifti.affine
        # print("Shape of the ground truth label: ", truth_matrix.shape)


        pred_list = [] # list of all the predictions


        for dropout in dropout_diffs.keys():
            pred_list_dropout = []

            for iter in range(10):
                # load label
                pred_path = os.path.join(dataset, dropout, 'predsTs_' + str(iter), label_name)
                pred_nifti = nib.load(pred_path)
                pred_matrix = pred_nifti.get_fdata() # 3D numpy matrix with an image
                pred_affinity = pred_nifti.affine
                assert (pred_affinity == affinity).all()
              
                pred_list_dropout.append(pred_matrix)


                ### method 1 (me) - for every image we calculate mean difference between prediction pixels and ground truth pixels ###
                # if method1:
                #   diff_label_iter = np.abs(truth_matrix - pred_matrix)
                #   mean_diff_iter = np.mean(np.mean(np.mean(diff_label_iter)))
                #   # print("mean_diff: ", mean_diff)
                #   diff_list_bayesian.append(mean_diff_iter)
                #   dropout_diffs[dropout].append(mean_diff_iter)
                  


            
            ##### predictions for a particular test image for a particular dropout #####
            pred_stack_dropout = np.stack(pred_list_dropout) # rónoważne do np.asarray w tym przypadku
            pred_list += pred_list_dropout
            # print("pred_stack_dropout.shape", pred_stack_dropout.shape)


            ### method 1 (me) - for every image we calculate mean difference between prediction pixels and ground truth pixels ###
            if method1:
              mean_label_dropout = np.mean(pred_stack_dropout, axis=0)
              assert mean_label_dropout.shape == truth_matrix.shape

              # var_dropout = np.var(pred_stack_dropout, axis=0)
              # assert var_dropout.shape == truth_matrix.shape

              nifti_mean_dropout = nib.Nifti1Image(mean_label_dropout, affine=affinity)
              nib.save(nifti_mean_dropout, os.path.join(dataset, dropout, "predsTs_average", label_name))

              diff_label_dropout = np.abs(truth_matrix - mean_label_dropout)
              mean_diff_dropout = np.mean(np.mean(np.mean(diff_label_dropout)))
              # print(mean_diff_dropout == np.mean(tmp), mean_diff_dropout, np.mean(tmp))
              dropout_means[dropout].append(mean_diff_dropout)



            ### method 2 (Z. Tabor) - how the prediction quality depends on the dropout rate ###
            if method2:
              mean_label_dropout = np.mean(pred_stack_dropout, axis=0)
              assert mean_label_dropout.shape == truth_matrix.shape

              # binarize
              bin_label_dropout = np.where(mean_label_dropout >= 0.5, 1, 0)

              # dice
              if np.sum(bin_label_dropout) + np.sum(truth_matrix) == 0: # całkowita zgodność, więc mamy 1
                dice_dropout = 1
              else:
                dice_dropout = np.sum(bin_label_dropout[truth_matrix==1])*2.0 / (np.sum(bin_label_dropout) + np.sum(truth_matrix))
              
              dropout_dices[dropout].append(dice_dropout)



            ### method 3 (Z. Tabor) - how the prediction uncertainty depends on the dropout rate ###
            if method3:
              std_label_dropout = np.std(pred_stack_dropout, axis=0)
              assert std_label_dropout.shape == truth_matrix.shape

              if number_of_files < 100:
                for std_pixel in std_label_dropout.flatten():
                  std_all_pixels.append(std_pixel)

              q95_dropout = np.quantile(std_label_dropout.flatten(), 0.95)
              q90_dropout = np.quantile(std_label_dropout.flatten(), 0.90)
              q75_dropout = np.quantile(std_label_dropout.flatten(), 0.75)
              q50_dropout = np.quantile(std_label_dropout.flatten(), 0.50)

              dropout_q95[dropout].append(q95_dropout)
              dropout_q90[dropout].append(q90_dropout)
              dropout_q75[dropout].append(q75_dropout)
              dropout_q50[dropout].append(q50_dropout)

              # # for visualization for an example label
              # if label_name == test_labels[0]:
              #     print(q95_dropout, q75_dropout, q50_dropout)
              #     plt.figure()
              #     sns.kdeplot(data=pd.DataFrame(std_label_dropout.flatten()), clip=[0, 0.0015], common_norm=False)
              #     # plt.xticks(list(plt.xticks()[0]) + [q95_dropout, q75_dropout, q50_dropout]) #, ["Q95", "Q75", "Q50"]
              #     plt.axvline(q95_dropout, color='b')
              #     plt.axvline(q75_dropout, color='b')
              #     plt.axvline(q50_dropout, color='b')
              #     plt.title("Density plot of standard deviation values for every pixel \n in every prediction for an example image with quantiles")
              #     plt.xlabel("Standard deviation")
              #     plt.ylabel("Density")
              #     plt.savefig(dataset + "/plots/example_quantiles.png")
              #     plt.close()




        ####### predictions for a particular test image #######
        pred_stack = np.stack(pred_list) # rónoważne do np.asarray w tym przypadku
        # print("pred_stack.shape", pred_stack.shape)


        ### method 1 (me) - for every image we calculate mean difference between prediction pixels and ground truth pixels ###
        if method1:
          mean_label = np.mean(pred_stack, axis=0)
          assert mean_label.shape == truth_matrix.shape

          # var_label = np.var(pred_stack, axis=0)
          # assert var_label.shape == truth_matrix.shape

          nifti_mean = nib.Nifti1Image(mean_label, affine=affinity)
          nib.save(nifti_mean, os.path.join(dataset, "predsTs_average", label_name))








  ################################# TRADITIONAL NETWORK #################################
  if method1 or method2:
    print("########### TRADITIONAL NETWORK ###########")
    diff_list_tr = []
    # dice_list_tr = []
    for label_name in test_labels:
        # print("-------", label_name, "-------")
        # load label
        truth_path = os.path.join(dataset, 'labelsTs', label_name)
        truth_nifti = nib.load(truth_path)
        truth_matrix = truth_nifti.get_fdata() # 3D numpy matrix with an image
        affinity = truth_nifti.affine
        # print("Shape of the ground truth label: ", truth_matrix.shape)

        # load label
        pred_path = os.path.join(dataset, 'DROPOUT00/predsTs', label_name)
        pred_nifti = nib.load(pred_path)
        pred_matrix = pred_nifti.get_fdata() # 3D numpy matrix with an image
        pred_affinity = pred_nifti.affine
        assert (pred_affinity == affinity).all()
      
        pred_list.append(pred_matrix)

        diff_label = np.abs(truth_matrix - pred_matrix)
        mean_diff = np.mean(np.mean(np.mean(diff_label)))
        # print("mean_diff: ", mean_diff)

        if method1:
          diff_list_tr.append(mean_diff)


        if method2:
          # binarize
          bin_label = np.where(pred_matrix >= 0.5, 1, 0)
          # dice
          if np.sum(bin_label) + np.sum(truth_matrix) == 0: # pełna zgodność, więc 1
            dice = 1
          else:
            dice = np.sum(bin_label[truth_matrix==1])*2.0 / (np.sum(bin_label) + np.sum(truth_matrix))

          dropout_dices["Traditional"].append(dice)








  #################################### PLOTS AND PRINTS ####################################
  dropout_dict = {"DROPOUT02": 0.2, "DROPOUT04": 0.4, "DROPOUT06": 0.6, "DROPOUT08": 0.8}

  ### my method - for every image we calculate mean difference between prediction pixels and ground truth pixels ###
  if method1:
    # for i in dropout_diffs.keys():
    #     plt.figure()
    #     plt.hist(dropout_diffs[i], bins=100)
    #     plt.title("Histogram of difference score between the ground truth images \n and the predictions for dropout = " + str(dropout_dict[i]))
    #     plt.xlabel("Difference score")
    #     plt.ylabel("Number of images")
    #     plt.savefig(dataset + "/plots/hist_" + i + ".png")
    #     plt.close()


    # plt.figure()
    # plt.hist(diff_list_bayesian, bins=100)
    # plt.title("Histogram of difference score between the ground truth images \n and the individual predictions (Bayesian network)")
    # plt.xlabel("Difference score")
    # plt.ylabel("Number of images")
    # plt.savefig(dataset + "/plots/hist_bayes.png")
    # plt.close()


    # # nie da się ustawić koloru :(
    # plt.figure()
    # sns.kdeplot(data=pd.DataFrame(diff_list_tr), clip=[0, 1], common_norm=False, palette="husl")
    # sns.kdeplot(data=pd.DataFrame.from_dict(dropout_diffs), clip=[0, 1], common_norm=False)
    # # legend = [i for i in dropout_scores_all.keys()]
    # # legend.append("Traditional")
    # plt.title("Density plot of difference score between the ground truth images \n and the individual predictions (Bayesian and traditional networks)")
    # plt.xlabel("Difference score")
    # plt.ylabel("Density")
    # # plt.legend(legend)
    # plt.savefig(dataset + "/plots/dist_indiv_trad_bayes.png")
    # plt.close()



    # dropout_means["Traditional"] = diff_list_tr
    # for el in dropout_means:
    #     print(len(dropout_means[el]))
    plt.figure()
    sns.kdeplot(data=pd.DataFrame.from_dict(dropout_means), clip=[0, 1], common_norm=False)
    plt.title("Density plot of difference score between the ground truth images \n and the mean predictions (Bayesian and traditional networks)")
    plt.xlabel("Difference score")
    plt.ylabel("Density")
    plt.savefig(dataset + "/plots/dist_mean_trad_bayes.png")
    plt.close()



    # plt.figure()
    # sns.kdeplot(data=pd.DataFrame(diff_list_bayesian), clip=[0, 1])
    # sns.kdeplot(data=pd.DataFrame(diff_list_tr), clip=[0, 1], common_norm=False)
    # plt.title("Density plot of difference score between the ground truth images \n and the individual predictions (Bayesian and traditional networks)")
    # plt.xlabel("Difference score")
    # plt.ylabel("Density")
    # plt.savefig(dataset + "/plots/dist_all_trad_bayes.png")
    # plt.close()




    # plt.figure()
    # plt.hist(diff_list_tr, bins=100)
    # plt.title("Histogram of difference score between the ground truth images \n and the individual predictions (traditional network)")
    # plt.xlabel("Difference score")
    # plt.ylabel("Number of images")
    # plt.savefig(dataset + "/plots/hist_trad.png")
    # plt.close()


    # plt.figure()
    # # diff_list = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    # sns.kdeplot(data=pd.DataFrame(diff_list_tr), clip=[0, 1], common_norm=False)
    # plt.title("Density plot of difference score between the ground truth images \n and the individual predictions (traditional network)")
    # plt.xlabel("Difference score")
    # plt.ylabel("Density")
    # plt.savefig(dataset + "/plots/dist_trad.png")
    # plt.close()






  ### method 2 (Z. Tabor) - how the prediction quality depends on the dropout rate  ###
  if method2:
    # dropout_dices["Traditional"] = dice_list_tr

    plt.figure()
    sns.kdeplot(data=pd.DataFrame(dropout_dices), clip=[0, 1], common_norm=False)
    plt.title("Density plot of Dice score between the ground truth images \n and the mean predictions (Bayesian and traditional networks)")
    plt.xlabel("Difference score")
    plt.ylabel("Density")
    plt.savefig(dataset + "/plots/dice_trad_bayes.png")
    plt.close()



    # print("Mean dice coefficient for different dropouts:")
    # # for dropout in dropout_dices.keys():
    # #     print(dropout, "\t", np.mean(dropout_dices[dropout]))

    dice_table_string += "\\textbf{" + dataset + "} & " + str(np.round(np.mean(dropout_dices["Traditional"]), 10)) + "\t"
    for dropout in dropout_dices.keys():
      val = np.round(np.mean(dropout_dices[dropout]), 10)
      if dropout == "Traditional":
        continue
      elif dropout == "DROPOUT08":
        dice_table_string += "& " + str(val) + "\t \\\\ \\hline\n"
      else:
        dice_table_string += "& \multicolumn{1}{l|}{" + str(val) + "}\t"

      print(np.min(dropout_dices[dropout]), np.max(dropout_dices[dropout]))





  ### method 3 (Z. Tabor)
  if method3:
    q95 = np.quantile(std_all_pixels, 0.95)
    q90 = np.quantile(std_all_pixels, 0.90)
    q75 = np.quantile(std_all_pixels, 0.75)
    q50 = np.quantile(std_all_pixels, 0.50)

    plt.figure()
    if dataset == "LUNG":
      thr = 0.025
    elif dataset == "KNEE":
      thr = 0.005
    elif dataset == "SKB":
      thr = 0.1
    sns.kdeplot(data=std_all_pixels, clip=[0, thr], common_norm=False)
    plt.title("Density plot of standard deviation (in the range of each dropout) \nfor all pixels")
    plt.axvline(q95, color='b')
    plt.text(q95, 250, "Q95", color='b')
    plt.axvline(q90, color='m')
    plt.text(q90, 200, "Q90", color='m')
    plt.axvline(q75, color='g')
    plt.text(q75, 150, "Q75", color='g')
    plt.axvline(q50, color='r') 
    plt.text(q50, 100, "Q50", color='r')
    plt.xlabel("Standard deviation")
    plt.ylabel("Density")
    plt.savefig(dataset + "/plots/std_all_pixels.png")
    plt.close()


    print("Means from quantile values of standard deviation of the predictions depending on dropout rate")
    # print("DROPOUT RATE \t QUANTILE 95% \t QUANTILE 75% \t QUANTILE 50%")
    # for dropout in dropout_dict.keys():
      # print(dropout, "\t", np.mean(dropout_q95[dropout]), "\t", np.mean(dropout_q75[dropout]), "\t", np.mean(dropout_q50[dropout]))


    q95_table_string += "\\textbf{" + dataset + "} "
    for dropout in dropout_dict.keys():
      val = '%.4g' % np.mean(dropout_q95[dropout])  # np.round(np.mean(dropout_q95[dropout]), 10)
      if dropout == "DROPOUT08":
        q95_table_string += "& " + val + "\t \\\\ \\hline \n"
      else:
        q95_table_string += "& \multicolumn{1}{l|}{" + val + "}\t"


    q90_table_string += "\\textbf{" + dataset + "} "
    for dropout in dropout_dict.keys():
      val = '%.4g' % np.mean(dropout_q90[dropout])  # np.round(np.mean(dropout_q90[dropout]), 10)
      if dropout == "DROPOUT08":
        q90_table_string += "& " + val + "\t \\\\ \\hline \n"
      else:
        q90_table_string += "& \multicolumn{1}{l|}{" + val + "}\t"


    q75_table_string += "\\textbf{" + dataset + "} "
    for dropout in dropout_dict.keys():
      val = '%.4g' % np.mean(dropout_q75[dropout])  # np.round(np.mean(dropout_q75[dropout]), 10)
      if dropout == "DROPOUT08":
        q75_table_string += "& " + val + "\t \\\\ \\hline \n"
      else:
        q75_table_string += "& \multicolumn{1}{l|}{" + val + "}\t"


    q50_table_string += "\\textbf{" + dataset + "} "
    for dropout in dropout_dict.keys():
      val = '%.4g' % np.mean(dropout_q50[dropout])  # np.round(np.mean(dropout_q50[dropout]), 10)
      if dropout == "DROPOUT08":
        q50_table_string += "& " + val + "\t \\\\ \\hline \n"
      else:
        q50_table_string += "& \multicolumn{1}{l|}{" + val + "}\t"


print("Dice")
print(dice_table_string)

print("Q95")
print(q95_table_string)
print("Q90")
print(q90_table_string)
print("Q75")
print(q75_table_string)
print("Q50")
print(q50_table_string)