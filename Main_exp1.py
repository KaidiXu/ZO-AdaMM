###  Application to find adversarial perturbation per image

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
matplotlib.rcParams.update({'font.size': 12})
from SubFunctions_blackBox_attack import main

### code to choose dataset
code = 3 ### 1 for MNIST; 2 for CIFAR-10; 3 for imagenet


### natural examples
IDs_CIFAR = [6,10,1,5,17,25,28,33,36,68] ### Example of images; can be extended to more specific images in image dataset
IDs_MNIST = [2,34,3,4,7,15,21,38,44,177]
IDs_imagenet = [2,34,3,4,7,15,21,38,44,177]  # 2000 choices

args = {} ### input parameters

if code == 1:
    IDs = IDs_MNIST.copy()
if code == 2:
    IDs = IDs_CIFAR.copy()
if code == 3:
    IDs = IDs_imagenet.copy()

flag_train = 1  ### flag to train network or not, 0: not to train, 1: to train
flag_plot = 0 ### flag to post-process data

# ### parameter setting for learning rate
lr_vec = [ 0.001 ] ### -4, constant learning rate, or constant/\sqrt{t}

### set common parameters
if code == 1:
    Imax = 800
    dataset = "mnist"
    args["maxiter"] = Imax + 0  ### max iterations of an algorithm
    args["init_const"] = 100  ### regularization parameter prior to attack loss
    args["dataset"] = "mnist"
elif code == 2:
    Imax = 800
    dataset = "cifar10"
    args["maxiter"] = Imax + 0
    args["init_const"] = 0.1  ### regularization parameter prior to attack loss
    args["dataset"] = "cifar10"
elif code == 3:
    Imax = 1600
    dataset = "imagenet"
    args["maxiter"] = Imax + 0
    args["init_const"] = 0.1  ### regularization parameter prior to attack loss
    args["dataset"] = "imagenet"

args["kappa"] = 1e-10  ### attack confidence level in attack loss
args["save_iteration"] = False
args["targeted_attack"] = False
args["print_iteration"] = True
args["decay_lr"] = True
args["exp_code"] = 5

### parameter setting for ZO gradient estimation
args["q"] = 10  ### number of random direction vectors
args["mu"] = 0.001  ### key parameter: smoothing parameter in ZO gradient estimation # 0.001 for imagenet

### parameter setting for mini-batch
args["mini_batch_sz"] = 1

### encode unconstrained or constrained optimization algorithms
alg_dic = {}
alg_dic['uncons'] = ['ZOSGD', 'ZOSCD', 'ZOsignSGD', 'ZOAdaMM']
alg_dic['cons'] = ['ZOSMD', 'ZOPSGD', 'ZONES', 'ZOAdaMM']
#alg_dic['cons'] = ['ZOAdaMM']

for idx_image in range(len(IDs)): ### per image ID

    image_id = IDs[idx_image]
    args["img_id"] = image_id


    for i_lr in range(len(lr_vec)): ### per learning rate
        lr_temp = lr_vec[i_lr]  ### key parameter I: learning rate
        args["lr_idx"] = i_lr

        ### parameter setting for learning rate
        args["lr"] = lr_temp

        # if (args["dataset"] == 'mnist'):  ########## learning rate
        #     args["lr"] = 0.05
        # elif (args["dataset"] == 'cifar10'):
        #     args["lr"] = 0.0005

        ### all algorithms
        for class_i in list(alg_dic.keys()):
            algs_class_i = alg_dic[class_i]
            args["constraint"] = class_i

            for j_alg in range(len(algs_class_i)):
                args["mode"] = algs_class_i[j_alg]
                if flag_train:
                    # if args["constraint"] == 'cons':
                        main(args)

    ########## plot region: post - processing
    if flag_plot:
        ### plots: per image ID
        mark_type_color = ['^', 's', 'o', 'd']
        color_list = ['black', 'tab:olive', 'tab:cyan',  'red', 'green', 'tab:orange', 'tab:pink', 'blue']
        line_list = ['-','-.']

        num_algorithm = int(len(alg_dic['uncons']) + len(alg_dic['cons']))

        total_loss_matrix = np.zeros((len(lr_vec),num_algorithm,Imax))
        distortion_loss_matrix = np.zeros((len(lr_vec),num_algorithm,Imax))
        attack_loss_matrix = np.zeros((len(lr_vec),num_algorithm,Imax))

        ### last iter attack
        distortion_last_matrix = np.zeros((len(lr_vec), num_algorithm)) + np.nan
        attack_loss_last_matrix = np.zeros((len(lr_vec), num_algorithm)) + np.nan


        ### first attack
        iter_first_matrix = np.zeros((len(lr_vec),num_algorithm)) + np.nan
        distortion_first_matrix = np.zeros((len(lr_vec),num_algorithm)) + np.nan


        ### best iter attack
        iter_best_matrix =  np.zeros((len(lr_vec),num_algorithm)) + np.nan
        distortion_best_matrix =  np.zeros((len(lr_vec),num_algorithm)) + np.nan


        ### plot: total loss versus iteration for all algorithms & all learn rates
        for i_lr in range(len(lr_vec)):  ### per learning rate
            # plt.figure(int(3*i_lr+1)) ### create a figure for attack loss
            # plt.figure(int(3*i_lr+2)) ### create a figure for distortion
            # plt.figure(int(3*i_lr+3)) ### create a figure for total loss

            fig1, axs1 = plt.subplots(1, 2, constrained_layout=True)
            fig2, axs2 = plt.subplots(1, 2, constrained_layout=True)
            fig3, axs3 = plt.subplots(1, 2, constrained_layout=True)


            idx_algorithm = -1
            legend_name = []
            for class_i in list(alg_dic.keys()):
                algs_class_i = alg_dic[class_i]
                args["constraint"] = class_i
                # if class_i == 'uncons':
                #     continue
                for j_alg in range(len(algs_class_i)):
                    args["mode"] = algs_class_i[j_alg]
                    idx_algorithm = idx_algorithm + 1
                    legend_name.append(args["mode"])
                    ### read stored document
                    suffix0 = "Results_SL/id_{}_Mode_{}_{}_lr_{}_decay_{}_case{}".format(image_id, args["mode"], args["constraint"], i_lr, int(args["decay_lr"]),args["exp_code"])
                    npzfile = np.load(suffix0 + ".npz")
                    attack_flag = npzfile['attack_flag']
                    total_loss = npzfile['loss']
                    loss_perturbation = npzfile['perturbation']
                    attack_loss = total_loss - loss_perturbation

                    total_loss_matrix[i_lr,idx_algorithm,:] = total_loss
                    distortion_loss_matrix[i_lr,idx_algorithm,:] = loss_perturbation
                    attack_loss_matrix[i_lr,idx_algorithm,:] = attack_loss

                    distortion_last_matrix[i_lr,idx_algorithm] = np.sqrt(loss_perturbation[-1])
                    attack_loss_last_matrix[i_lr,idx_algorithm] = attack_loss[-1]

                    if args["constraint"] == 'cons':
                        legend_name[-1] = args["mode"] + '_cons'

                    if attack_flag:  ### attack succeeds

                        iter_first_attack = npzfile['first_iteration']
                        iter_first_matrix[i_lr,idx_algorithm] = iter_first_attack

                        distortion_first_attack = npzfile['first_distortion']
                        distortion_first_matrix[i_lr,idx_algorithm] = distortion_first_attack

                        iter_best_attack = npzfile['best_iteation']
                        iter_best_matrix[i_lr,idx_algorithm] = iter_best_attack

                        distortion_best_attack = npzfile['best_distortion']
                        distortion_best_matrix[i_lr,idx_algorithm] = distortion_best_attack

                        ### plot
                        if args["constraint"] == 'cons':
                            # plt.figure(int(3 * i_lr + 1))
                            axs1[1].plot(attack_loss, color=color_list[idx_algorithm], linestyle=line_list[1], label = legend_name[-1])
                            # plt.figure(int(3 * i_lr + 2))
                            axs2[1].plot(np.sqrt(loss_perturbation), color=color_list[idx_algorithm], linestyle=line_list[1],
                                     label=legend_name[-1])
                            # plt.figure(int(3 * i_lr + 3))
                            axs3[1].plot(total_loss, color=color_list[idx_algorithm], linestyle=line_list[1], label = legend_name[-1])
                        else:
                            # plt.figure(int(3 * i_lr + 1))
                            axs1[0].plot(attack_loss, color=color_list[idx_algorithm], linestyle=line_list[0], label = legend_name[-1])
                            # plt.figure(int(3 * i_lr + 2))
                            axs2[0].plot(np.sqrt(loss_perturbation), color=color_list[idx_algorithm], linestyle=line_list[0],
                                     label=legend_name[-1])
                            # plt.figure(int(3 * i_lr + 3))
                            axs3[0].plot(total_loss, color=color_list[idx_algorithm], linestyle=line_list[0], label = legend_name[-1])

                        if args["constraint"] == 'cons':
                            # plt.figure(int(3 * i_lr + 1))
                            axs1[1].plot(iter_first_attack, attack_loss[iter_first_attack],color=color_list[idx_algorithm], marker='o', linewidth=2, markersize=12, label = None)

                            # plt.figure(int(3 * i_lr + 2))
                            axs2[1].plot(iter_first_attack, np.sqrt(loss_perturbation[iter_first_attack]),color=color_list[idx_algorithm], marker='o', linewidth=2, markersize=12, label = None)

                            # plt.figure(int(3 * i_lr + 1))
                            axs3[1].plot(iter_first_attack, total_loss[iter_first_attack],color=color_list[idx_algorithm], marker='o', linewidth=2, markersize=12, label = None)
                        else:
                            # plt.figure(int(3 * i_lr + 1))
                            axs1[0].plot(iter_first_attack, attack_loss[iter_first_attack],color=color_list[idx_algorithm], marker='s', linewidth=2, markersize=12, label = None)
                            # plt.figure(int(3 * i_lr + 2))
                            axs2[0].plot(iter_first_attack, np.sqrt(loss_perturbation[iter_first_attack]),color=color_list[idx_algorithm], marker='s', linewidth=2, markersize=12, label = None)
                            # plt.figure(int(3 * i_lr + 3))
                            axs3[0].plot(iter_first_attack, total_loss[iter_first_attack],color=color_list[idx_algorithm], marker='s', linewidth=2, markersize=12, label = None)
                    else:
                        ### plot
                        if args["constraint"] == 'cons':
                            # plt.figure(int(3 * i_lr + 1))
                            axs1[1].plot(attack_loss, color=color_list[idx_algorithm], linestyle=line_list[1], label = legend_name[-1])
                            # plt.figure(int(3 * i_lr + 2))
                            axs2[1].plot(np.sqrt(loss_perturbation), color=color_list[idx_algorithm], linestyle=line_list[1],
                                     label=legend_name[-1])
                            # plt.figure(int(3 * i_lr + 3))
                            axs3[1].plot(total_loss, color=color_list[idx_algorithm], linestyle=line_list[1], label = legend_name[-1])

                        else:
                            # plt.figure(int(3 * i_lr + 1))
                            axs1[0].plot(attack_loss, color=color_list[idx_algorithm], linestyle=line_list[0], label = legend_name[-1])
                            # plt.figure(int(3 * i_lr + 2))
                            axs2[0].plot(np.sqrt(loss_perturbation), color=color_list[idx_algorithm], linestyle=line_list[0],
                                     label=legend_name[-1])
                            # plt.figure(int(3 * i_lr + 3))
                            axs3[0].plot(total_loss, color=color_list[idx_algorithm], linestyle=line_list[0], label = legend_name[-1])

            ### finish plotting
            # plt.figure(int(3*i_lr + 1))
            axs1[0].legend(loc="best",
                        #  bbox_to_anchor=(0.65, 1.1),
                       ncol=1)
            axs1[0].set_xlabel("Iteration")
            axs1[0].set_ylabel("Attack loss")
            axs1[1].legend(loc="best",
                        #  bbox_to_anchor=(0.65, 1.1),
                       ncol=1)
            axs1[1].set_xlabel("Iteration")
            axs1[1].set_ylabel("Attack loss")
            plt.show( )
            plt.pause(0.5)
            suffix_plot = "id_{}_lr_{}_LossPlot_case{}".format(image_id,i_lr,args["exp_code"] )
            fig1.savefig("Plots_SL/{}.png".format(suffix_plot))
            plt.close(fig1)
            #
            # plt.figure(int(3*i_lr + 2))
            axs2[0].legend(loc="best",
                        #  bbox_to_anchor=(0.65, 1.1),
                           ncol=1)
            axs2[0].set_xlabel("Iteration")
            axs2[0].set_ylabel("Distortion")
            axs2[1].legend(loc="best",
                        #  bbox_to_anchor=(0.65, 1.1),
                           ncol=1)
            axs2[1].set_xlabel("Iteration")
            axs2[1].set_ylabel("Distortion")
            plt.show( )
            plt.pause(0.5)
            suffix_plot = "id_{}_lr_{}_DistortionPlot_case{}".format(image_id,i_lr,args["exp_code"] )
            fig2.savefig("Plots_SL/{}.png".format(suffix_plot))
            plt.close(fig2)


            # total loss convergence
            axs3[0].legend(loc="best",
                        #  bbox_to_anchor=(0.65, 1.1),
                           ncol=1)
            axs3[0].set_xlabel("Iteration")
            axs3[0].set_ylabel("Objective value")
            axs3[1].legend(loc="best",
                        #  bbox_to_anchor=(0.65, 1.1),
                           ncol=1)
            axs3[1].set_xlabel("Iteration")
            axs3[1].set_ylabel("Objective value")
            # plt.tight_layout()
            plt.show( )
            plt.pause(0.5)
            suffix_plot = "id_{}_lr_{}_ObjectValPlot_case{}".format(image_id,i_lr,args["exp_code"] )
            fig3.savefig("Plots_SL/{}.png".format(suffix_plot))
            plt.close(fig3)

        # ### plot heatmap distortion_loss_last_matrix for all lrs
        # fig, axs = plt.subplots(constrained_layout=True)
        # cmap = matplotlib.cm.jet
        # cmap.set_bad('white', 1.)
        #
        # ### distortion last
        # cs0 = axs.imshow(np.transpose((distortion_last_matrix)), interpolation='nearest', cmap=cmap)
        # fig.colorbar(cs0, ax=axs, shrink=0.9)
        # labels_string_x = [str("%2.2f" % item) for item in np.log10(lr_vec)]
        # labels_string_y = legend_name
        # plt.xticks(range(np.size((distortion_last_matrix), 0)), labels_string_x)
        # plt.xticks(rotation=90)
        # plt.yticks(range(np.size((distortion_last_matrix), 1)), labels_string_y)
        # plt.xlabel("Learning rate (log10)")
        # plt.ylabel("ZO method")
        # plt.title('Distortion_final_attack')
        # suffix_plot = "id_{}_final_distortion_case{}".format(image_id,args["exp_code"] )
        # plt.pause(1.5)
        # plt.savefig("Plots_SL/{}.png".format(suffix_plot))
        # plt.close()
        #
        #
        # # ### plot distortion_first_matrix
        # #
        # # fig, axs = plt.subplots(constrained_layout=True)
        # # cmap = matplotlib.cm.jet
        # # cmap.set_bad('white', 1.)
        # #
        # # cs0 = axs.imshow(np.transpose(distortion_first_matrix), interpolation='nearest', cmap=cmap)
        # # fig.colorbar(cs0, ax=axs, shrink=0.9)
        # # labels_string_x = [str("%2.2f" % item) for item in np.log10(lr_vec)]
        # # labels_string_y = legend_name
        # # plt.xticks(range(np.size((distortion_first_matrix), 0)), labels_string_x)
        # # plt.xticks(rotation=90)
        # # plt.yticks(range(np.size((distortion_first_matrix), 1)), labels_string_y)
        # # plt.xlabel("Learning rate (log10)")
        # # plt.ylabel("ZO method")
        # # plt.title('Distortion_first_attack')
        # # suffix_plot = "id_{}_first_att_distortion_case{}".format(image_id,args["exp_code"] )
        # # plt.pause(1.5)
        # # plt.savefig("Plots_SL/{}.png".format(suffix_plot))
        # # plt.close()
        #
        # ### plot distortion_best_matrix
        # fig, axs = plt.subplots(constrained_layout=True)
        # cmap = matplotlib.cm.jet
        # cmap.set_bad('white', 1.)
        #
        # cs0 = axs.imshow(np.transpose(distortion_best_matrix), interpolation='nearest', cmap=cmap)
        # fig.colorbar(cs0, ax=axs, shrink=0.9)
        # labels_string_x = [str("%2.2f" % item) for item in np.log10(lr_vec)]
        # labels_string_y = legend_name
        # plt.xticks(range(np.size((distortion_best_matrix), 0)), labels_string_x)
        # plt.xticks(rotation=90)
        # plt.yticks(range(np.size((distortion_best_matrix), 1)), labels_string_y)
        # plt.xlabel("Learning rate (log10)")
        # plt.ylabel("ZO method")
        # plt.title('Distortion_best_attack')
        # suffix_plot = "id_{}_best_att_distortion_case{}".format(image_id,args["exp_code"] )
        # plt.pause(1.5)
        # plt.savefig("Plots_SL/{}.png".format(suffix_plot))
        # plt.close()