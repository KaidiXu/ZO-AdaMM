# %load_ext autoreload
# %autoreload 2
import os
import sys
import tensorflow as tf
import numpy as np
import random
import time
from keras.layers import Lambda
from tqdm import tqdm
from setup_mnist import MNIST, MNISTModel
from setup_cifar import CIFAR, CIFARModel
from setup_inception import ImageNet, InceptionModel

import Utils as util

SEED = 121

args = {} 
args["maxiter"] = 1000 + 0
args["init_const"] = 10  ### regularization parameter prior to attack loss
args["dataset"] = "imagenet"

args["kappa"] = 1e-10  ### attack confidence level in attack loss
args["save_iteration"] = False
args["targeted_attack"] = False
args["print_iteration"] = True 
args["decay_lr"] = True
args["exp_code"] = 50

### parameter setting for ZO gradient estimation
args["q"] = 10  ### number of random direction vectors
args["mu"] = 0.005  ### key parameter: smoothing parameter in ZO gradient estimation # 0.001 for imagenet

### parameter setting for mini-batch
args["mini_batch_sz"] = 1
args["img_id"] = 11
args["target_id"] = 23
args["image_number"] = 50

args["lr_idx"] = 0
args["lr"] = 3e-4  #0.02 for uncons
args["constraint"] = 'cons' #'uncons'
args["mode"] = "ZOSMD" 
#alg_dic['uncons'] = ['ZOSGD', 'ZOSCD', 'ZOsignSGD', 'ZOAdaMM']
#alg_dic['cons'] = ['ZOSMD', 'ZOPSGD', 'ZONES', 'ZOAdaMM']

def main(args):
    with tf.Session() as sess:
        
        random.seed(SEED)
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        
        image_id_set = np.random.choice(range(1000), args["image_number"]*3, replace=False)
        #image_id_set = np.random.randint(1, 1000, args["image_number"] )
        arg_max_iter = args['maxiter'] ### max number of iterations
        arg_init_const = args['init_const'] ### regularization prior to attack loss
        arg_kappa = args['kappa'] ### attack confidence level
        arg_q = args['q'] ### number of random direction vectors
        arg_mode = args['mode'] ### algorithm name
        arg_save_iteration = args['save_iteration']
        arg_Dataset = args["dataset"]
        arg_targeted_attack = args["targeted_attack"]
        arg_bsz = args["mini_batch_sz"]
        idx_lr = args["lr_idx"]

        ## load classofier For MNIST and CIFAR pixel value range is [-0.5,0.5]
        if (arg_Dataset == 'mnist'):
            data, model = MNIST(), MNISTModel("models/mnist", sess, True)
        elif (arg_Dataset == 'cifar10'):
            data, model = CIFAR(), CIFARModel("models/cifar", sess, True)
        elif (arg_Dataset == 'imagenet'):
            data, model = ImageNet(SEED), InceptionModel(sess, True)
        else:
            print('Please specify a valid dataset')
        
        
        succ_count, ii, iii = 0, 0, 0
        final_distortion_count,first_iteration_count, first_distortion_count = [], [], []
        while iii < args["image_number"]:
            ii = ii + 1
            image_id = image_id_set[ii]
            
            if image_id!= 836: continue # for test only
            
            orig_prob, orig_class, orig_prob_str = util.model_prediction(model,
                                        np.expand_dims(data.test_data[image_id], axis=0)) ## orig_class: predicted label;
    
            if arg_targeted_attack: ### target attack
                target_label = np.remainder(orig_class + 1, 10)
            else:
                target_label = orig_class
    
            orig_img, target = util.generate_data(data, image_id, target_label)
            # shape of orig_img is (1,28,28,1) in [-0.5, 0.5]
    
            true_label_list = np.argmax(data.test_labels, axis=1)
            true_label = true_label_list[image_id]
    
    
            print("Image ID:{}, infer label:{}, true label:{}".format(image_id, orig_class, true_label))
            if true_label != orig_class:
                print( "True Label is different from the original prediction, pass!")
                continue
            else:
                iii = iii + 1
                
            print('\n', iii, '/',args["image_number"])
    
            ##  parameter
            d = orig_img.size  # feature dim
            print("dimension = ", d)
    
            # mu=1/d**2  # smoothing parameter
            q = arg_q + 0
            I = arg_max_iter + 0
            kappa = arg_kappa + 0
            const = arg_init_const + 0
    
    
            ## flatten image to vec
            orig_img_vec = np.resize(orig_img, (1, d))
            delta_adv = np.zeros((1,d)) ### initialized adv. perturbation
            #delta_adv = np.random.uniform(-16/255,16/255,(1,d))
        
            ## w adv image initialization
            if args["constraint"] == 'uncons':
                # * 0.999999 to avoid +-0.5 return +-infinity 
                w_ori_img_vec = np.arctanh(2 * (orig_img_vec) * 0.999999)  # in real value, note that orig_img_vec in [-0.5, 0.5]
                w_img_vec = np.arctanh(2 * (np.clip(orig_img_vec + delta_adv,-0.5,0.5)) * 0.999999) 
            else:
                w_ori_img_vec = orig_img_vec.copy()
                w_img_vec = np.clip(w_ori_img_vec + delta_adv,-0.5,0.5)
        
            # ## test ##
            # for test_value in w_ori_img_vec[0, :]:
            #     if np.isnan(test_value) or np.isinf(test_value):
            #         print(test_value)
    
    
            # initialize the best solution & best loss
            best_adv_img = []  # successful adv image in [-0.5, 0.5]
            best_delta = []    # best perturbation
            best_distortion = (0.5 * d) ** 2 # threshold for best perturbation
            total_loss = np.zeros(I) ## I: max iters
            l2s_loss_all = np.zeros(I)
            attack_flag = False
            first_flag = True ## record first successful attack
    
            # parameter setting for ZO gradient estimation
            mu = args["mu"] ### smoothing parameter
    
            ## learning rate
            base_lr = args["lr"]
    
            if arg_mode == "ZOAdaMM":
                ## parameter initialization for AdaMM
                v_init = 1e-7 #0.00001
                v_hat = v_init * np.ones((1, d))
                v = v_init * np.ones((1, d))
    
                m = np.zeros((1, d))
                # momentum parameter for first and second order moment
                beta_1 = 0.9
                beta_2 = 0.9  # only used by AMSGrad
                print(beta_1, beta_2)
    
            #for i in tqdm(range(I)):
            for i in range(I):
    
                if args["decay_lr"]:
                    base_lr = args["lr"]/np.sqrt(i+1)
    
                ## Total loss evaluation
                if args["constraint"] == 'uncons':
                    total_loss[i], l2s_loss_all[i] = function_evaluation_uncons(w_img_vec, kappa, target_label, const, model, orig_img,
                                                    arg_targeted_attack)
    
                else:
                    total_loss[i], l2s_loss_all[i] = function_evaluation_cons(w_img_vec, kappa, target_label, const, model, orig_img,
                                                               arg_targeted_attack)
    
    
    
                ## gradient estimation w.r.t. w_img_vec
                if arg_mode == "ZOSCD":
                    grad_est = grad_coord_estimation(mu, q, w_img_vec, d, kappa, target_label, const, model, orig_img,
                                                     arg_targeted_attack, args["constraint"])
                elif arg_mode == "ZONES":
                    grad_est = gradient_estimation_NES(mu, q, w_img_vec, d, kappa, target_label, const, model, orig_img,
                                                     arg_targeted_attack, args["constraint"])
                else:
                    grad_est = gradient_estimation_v2(mu, q, w_img_vec, d, kappa, target_label, const, model, orig_img,
                                                       arg_targeted_attack, args["constraint"])
    
                # if np.remainder(i,50)==0:
                # print("total loss:",total_loss[i])
                # print(np.linalg.norm(grad_est, np.inf))
    
                ## ZO-Attack, unconstrained optimization formulation
                if arg_mode == "ZOSGD":
                    delta_adv = delta_adv - base_lr * grad_est
                if arg_mode == "ZOsignSGD":
                    delta_adv = delta_adv - base_lr * np.sign(grad_est)
                if arg_mode == "ZOSCD":
                    delta_adv = delta_adv - base_lr * grad_est
                if arg_mode == "ZOAdaMM":
                    m = beta_1 * m + (1-beta_1) * grad_est
                    v = beta_2 * v + (1 - beta_2) * np.square(grad_est) ### vt
                    v_hat = np.maximum(v_hat,v)
                    #print(np.mean(v_hat))
                    delta_adv = delta_adv - base_lr * m /np.sqrt(v_hat)
                    if args["constraint"] == 'cons':
                        tmp = delta_adv.copy()
                        #X_temp = orig_img_vec.reshape((-1,1))
                        #V_temp2 = np.diag(np.sqrt(v_hat.reshape(-1)+1e-10))
                        V_temp = np.sqrt(v_hat.reshape(1,-1))
                        delta_adv = projection_box(tmp, orig_img_vec, V_temp, -0.5, 0.5)
                        #delta_adv2 = projection_box_2(tmp, X_temp, V_temp2, -0.5, 0.5)
                    # v_init = 1e-2 #0.00001
                    # v = v_init * np.ones((1, d))
                    # m = np.zeros((1, d))
                    # # momentum parameter for first and second order moment
                    # beta_1 = 0.9
                    # beta_2 = 0.99  # only used by AMSGrad
                    # m = beta_1 * m + (1-beta_1) * grad_est
                    # v = np.maximum(beta_2 * v + (1-beta_2) * np.square(grad_est),v)
                    # delta_adv = delta_adv - base_lr * m /np.sqrt(v+1e-10)
                    # if args["constraint"] == 'cons':
                    #     V_temp = np.diag(np.sqrt(v.reshape(-1)+1e-10))
                    #     X_temp = orig_img_vec.reshape((-1,1))
                    #     delta_adv = projection_box(delta_adv, X_temp, V_temp, -0.5, 0.5)
                if arg_mode == "ZOSMD":
                    delta_adv = delta_adv - 0.5*base_lr * grad_est
                    # delta_adv = delta_adv - base_lr* grad_est
                    if args["constraint"] == 'cons':
                        #V_temp = np.eye(orig_img_vec.size)
                        V_temp = np.ones_like(orig_img_vec)
                        #X_temp = orig_img_vec.reshape((-1,1))
                        delta_adv = projection_box(delta_adv, orig_img_vec, V_temp, -0.5, 0.5)
                if arg_mode == "ZOPSGD":
                    delta_adv = delta_adv - base_lr * grad_est
                    if args["constraint"] == 'cons':
                        #V_temp = np.eye(orig_img_vec.size)
                        V_temp = np.ones_like(orig_img_vec)
                        #X_temp = orig_img_vec.reshape((-1,1))
                        delta_adv = projection_box(delta_adv, orig_img_vec, V_temp, -0.5, 0.5)
                if arg_mode == "ZONES":
                    delta_adv = delta_adv - base_lr * np.sign(grad_est)
                    if args["constraint"] == 'cons':
                        #V_temp = np.eye(orig_img_vec.size)
                        V_temp = np.ones_like(orig_img_vec)
                        #X = orig_img_vec.reshape((-1,1))
                        delta_adv = projection_box(delta_adv, orig_img_vec, V_temp, -0.5, 0.5)
    
                # if arg_mode == "ZO-AdaFom":
                #     m = beta_1 * m + (1-beta_1) * grad_est
                #     v = v* (float(i)/(i+1)) + np.square(grad_est)/(i+1)
                #     w_img_vec = w_img_vec - base_lr * m/np.sqrt(v)
                ##
    
                ### adv. example update
                w_img_vec = w_ori_img_vec + delta_adv
    
    
                ## covert back to adv_img in [-0.5 , 0.5]
                if args["constraint"] == 'uncons':
                    adv_img_vec = 0.5 * np.tanh((w_img_vec)) / 0.999999 # 
                else:
                    adv_img_vec = w_img_vec.copy()
    
                adv_img = np.resize(adv_img_vec, orig_img.shape)
    
                ## update the best solution in the iterations
                attack_prob, _, _ = util.model_prediction(model, adv_img)
                target_prob = attack_prob[0, target_label]
                attack_prob_tmp = attack_prob.copy()
                attack_prob_tmp[0, target_label] = 0
                other_prob = np.amax(attack_prob_tmp)
    
    
                if args["print_iteration"]:
                    if np.remainder(i + 1, 1) == 0:
                        if true_label != np.argmax(attack_prob):
                            print("Iter %d (Succ): ID = %d, lr = %3.5f, decay = %d, ZO = %s %s, loss = %3.5f, l2sdist = %3.5f, TL = %d, PL = %d" % (i+1,
                                  image_id, args["lr"], int(args["decay_lr"]), arg_mode, args["constraint"], total_loss[i], l2s_loss_all[i], true_label, np.argmax(attack_prob)))
                        else:
                            print("Iter %d (Fail): ID = %d, lr = %3.6f, decay = %d, ZO = %s %s, loss = %3.5f, l2sdist = %3.5f, TL = %d, PL = %d" % (i + 1,
                                  image_id, args["lr"],  int(args["decay_lr"]), arg_mode, args["constraint"], total_loss[i], l2s_loss_all[i], true_label, np.argmax(attack_prob)))
    
    
    
                if arg_save_iteration:
                    os.system("mkdir Examples")
                    if (np.logical_or(true_label != np.argmax(attack_prob), np.remainder(i + 1, 10) == 0)): ## every 10 iterations
                        suffix = "id_{}_Mode_{}_True_{}_Pred_{}_Ite_{}".format(image_id, arg_mode, true_label,
                                                                               np.argmax(attack_prob), i + 1)
                        # util.save_img(adv_img, "Examples/{}.png".format(suffix))
    
                if arg_targeted_attack:
                    if (np.log(target_prob + 1e-10) - np.log(other_prob + 1e-10) >= kappa):  # check attack confidence
                        if (distortion(adv_img, orig_img) < best_distortion):  # check distortion
                            # print('best distortion obtained at',i,'-th iteration')
                            best_adv_img = adv_img
                            best_distortion = distortion(adv_img, orig_img)
                            best_delta = adv_img - orig_img
                            best_iteration = i + 1
                            adv_class = np.argmax(attack_prob)
                            attack_flag = True
                            ## Record first attack
                            if (first_flag):
                                first_flag = False  ### once gets into this, it will no longer record the next sucessful attack
                                first_adv_img = adv_img
                                first_distortion = distortion(adv_img, orig_img)
                                first_delta = adv_img - orig_img
                                first_class = adv_class
                                first_iteration = i + 1
                else:
                    if (np.log(other_prob + 1e-10) - np.log(target_prob + 1e-10) >= kappa):  # check attack confidence
                        if (distortion(adv_img, orig_img) < best_distortion):  # check distortion
                            # print('best distortion obtained at',i,'-th iteration')
                            best_adv_img = adv_img
                            best_distortion = distortion(adv_img, orig_img)
                            best_delta = adv_img - orig_img
                            best_iteration = i + 1
                            adv_class = np.argmax(attack_prob)
                            attack_flag = True
                            ## Record first attack
                            if (first_flag):
                                first_flag = False
                                first_adv_img = adv_img
                                first_distortion = distortion(adv_img, orig_img)
                                first_delta = adv_img - orig_img
                                first_class = adv_class
                                first_iteration = i + 1
    
            if (attack_flag):
                # os.system("mkdir Results_SL")
                # ## best attack (final attack)
                # suffix = "id_{}_Mode_{}_True_{}_Pred_{}".format(image_id, arg_mode, true_label, orig_class) ## orig_class, predicted label
                # suffix2 = "id_{}_Mode_{}_True_{}_Pred_{}".format(image_id, arg_mode, true_label, adv_class)
                # suffix3 = "id_{}_Mode_{}".format(image_id, arg_mode)
                # ### save original image
                # util.save_img(orig_img, "Results_SL/id_{}.png".format(image_id))
                # util.save_img(orig_img, "Results_SL/{}_Orig.png".format(suffix))
                # ### adv. image
                # util.save_img(best_adv_img, "Results_SL/{}_Adv_best.png".format(suffix2))
                # ### adv. perturbation
                # util.save_img(best_delta, "Results_SL/{}_Delta_best.png".format(suffix3))
                #
                #
                # ## first attack
                # suffix4 = "id_{}_Mode_{}_True_{}_Pred_{}".format(image_id, arg_mode, true_label, first_class)
                # ## first adv. imag
                # util.save_img(first_adv_img, "Results_SL/{}_Adv_first.png".format(suffix4))
                # ### first adv. perturbation
                # util.save_img(first_delta, "Results_SL/{}_Delta_first.png".format(suffix3))
    
                ## save data
                succ_count = succ_count + 1 
                final_distortion_count.append(l2s_loss_all[-1])
                first_distortion_count.append(first_distortion)
                first_iteration_count.append(first_iteration)
                suffix0 = "retperimage2/id_{}_Mode_{}_{}_lr_{}_decay_{}_case{}_per".format(image_id, arg_mode, args["constraint"], args["lr"], int(args["decay_lr"]), args["exp_code"] )
                np.savez("{}".format(suffix0), id=image_id, mode=arg_mode, loss=total_loss, perturbation=l2s_loss_all,
                         best_distortion=best_distortion, first_distortion=first_distortion,
                         first_iteration=first_iteration, best_iteation=best_iteration,
                         learn_rate=args["lr"], decay_lr = args["decay_lr"], attack_flag = attack_flag)
                ## print
                print("It takes {} iteations to find the first attack".format(first_iteration))
                # print(total_loss)
            else:
                ## save data
                suffix0 = "retperimage2/id_{}_Mode_{}_{}_lr_{}_decay_{}_case{}_per".format(image_id, arg_mode, args["constraint"], args["lr"], int(args["decay_lr"]), args["exp_code"] )
                np.savez("{}".format(suffix0), id=image_id, mode=arg_mode, loss=total_loss, perturbation=l2s_loss_all,
                         best_distortion=best_distortion,  learn_rate=args["lr"], decay_lr = args["decay_lr"], attack_flag = attack_flag)
                print("Attack Fails")
    
            sys.stdout.flush()
    print('succ rate:', succ_count/args["image_number"])
    print('average first success l2', np.mean(first_distortion_count))
    print('average first itrs', np.mean(first_iteration_count))
    print('average l2:', np.mean(final_distortion_count), ' best l2:', np.min(final_distortion_count), ' worst l2:', np.max(final_distortion_count))
    
    
# f: objection function
def function_evaluation(x, kappa, target_label, const, model, orig_img, arg_targeted_attack):
    # x is img_vec format in real value: w
    img_vec = 0.5 * np.tanh(x)/ 0.999999
    img = np.resize(img_vec, orig_img.shape)
    orig_prob, orig_class, orig_prob_str = util.model_prediction(model, img)
    tmp = orig_prob.copy()
    tmp[0, target_label] = 0
    if arg_targeted_attack:  # targeted attack
        Loss1 = const * np.max([np.log(np.amax(tmp) + 1e-10) - np.log(orig_prob[0, target_label] + 1e-10), -kappa])
    else:  # untargeted attack
        Loss1 = const * np.max([np.log(orig_prob[0, target_label] + 1e-10) - np.log(np.amax(tmp) + 1e-10), -kappa])

    Loss2 = np.linalg.norm(img - orig_img) ** 2
    return Loss1 + Loss2

# f: objection function for unconstrained optimization formulation
def function_evaluation_uncons(x, kappa, target_label, const, model, orig_img, arg_targeted_attack):
    # x in real value (unconstrained form), img_vec is in [-0.5, 0.5]
    img_vec = 0.5 * np.tanh(x) / 0.999999
    img = np.resize(img_vec, orig_img.shape)
    orig_prob, orig_class, orig_prob_str = util.model_prediction(model, img)
    tmp = orig_prob.copy()
    tmp[0, target_label] = 0
    if arg_targeted_attack:  # targeted attack, target_label is false label
        Loss1 = const * np.max([np.log(np.amax(tmp) + 1e-10) - np.log(orig_prob[0, target_label] + 1e-10), -kappa])
    else:  # untargeted attack, target_label is true label
        Loss1 = const * np.max([np.log(orig_prob[0, target_label] + 1e-10) - np.log(np.amax(tmp) + 1e-10), -kappa])

    Loss2 = np.linalg.norm(img - orig_img) ** 2
    return Loss1 + Loss2, Loss2

# f: objection function for constrained optimization formulation
def function_evaluation_cons(x, kappa, target_label, const, model, orig_img, arg_targeted_attack):
    # x is in [-0.5, 0.5]
    img_vec = x.copy()
    img = np.resize(img_vec, orig_img.shape)
    orig_prob, orig_class, orig_prob_str = util.model_prediction(model, img)
    tmp = orig_prob.copy()
    tmp[0, target_label] = 0
    if arg_targeted_attack:  # targeted attack, target_label is false label
        Loss1 = const * np.max([np.log(np.amax(tmp) + 1e-10) - np.log(orig_prob[0, target_label] + 1e-10), -kappa])
    else:  # untargeted attack, target_label is true label
        Loss1 = const * np.max([np.log(orig_prob[0, target_label] + 1e-10) - np.log(np.amax(tmp) + 1e-10), -kappa])

    Loss2 = np.linalg.norm(img - orig_img) ** 2 ### squared norm
    return Loss1 + Loss2, Loss2

# Elastic-net norm computation: L2 norm + beta * L1 norm
def distortion(a, b):
    return np.linalg.norm(a - b) ### square root


# random directional gradient estimation - averaged over q random directions
def gradient_estimation(mu,q,x,d,kappa,target_label,const,model,orig_img,arg_mode,arg_targeted_attack):
    # x is img_vec format in real value: w
    m, sigma = 0, 100 # mean and standard deviation
    f_0=function_evaluation(x,kappa,target_label,const,model,orig_img,arg_targeted_attack)
    grad_est=0
    for i in range(q):
        u = np.random.normal(m, sigma, (1,d))
        u_norm = np.linalg.norm(u)
        u = u/u_norm
        f_tmp=function_evaluation(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
        # gradient estimate
        if arg_mode == "ZO-M-signSGD":
            grad_est=grad_est+ np.sign(u*(f_tmp-f_0))
        else:
            grad_est=grad_est+ (d/q)*u*(f_tmp-f_0)/mu
    return grad_est
    #grad_est=grad_est.reshape(q,d)
    #return d*grad_est.sum(axis=0)/q

def gradient_estimation_v2(mu,q,x,d,kappa,target_label,const,model,orig_img,arg_targeted_attack,arg_cons):
    # x is img_vec format in real value: w
    # m, sigma = 0, 100 # mean and standard deviation
    sigma = 100
    # ## generate random direction vectors
    # U_all_new = np.random.multivariate_normal(np.zeros(d), np.diag(sigma*np.ones(d) + 0), (q,1))


    if arg_cons == 'uncons':
        f_0, ignore =function_evaluation_uncons(x,kappa,target_label,const,model,orig_img,arg_targeted_attack)
    else:
        f_0, ignore =function_evaluation_cons(x,kappa,target_label,const,model,orig_img,arg_targeted_attack)

    grad_est=0
    for i in range(q):
        u = np.random.normal(0, sigma, (1,d))
        u_norm = np.linalg.norm(u)
        u = u/u_norm
        # ui = U_all_new[i, 0].reshape(-1)
        # u = ui / np.linalg.norm(ui)
        # u = np.resize(u, x.shape)
        if arg_cons == 'uncons':
            ### x+mu*u = x0 + delta + mu*u: unconstrained in R^d, constrained in [-0.5,0.5]^d
            f_tmp, ignore = function_evaluation_uncons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
        else:
            f_tmp, ignore = function_evaluation_cons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
        # gradient estimate
        # if arg_mode == "ZO-M-signSGD":
        #     grad_est=grad_est+ np.sign(u*(f_tmp-f_0))
        # else:
        grad_est=grad_est+ (d/q)*u*(f_tmp-f_0)/mu
    return grad_est
    #grad_est=grad_est.reshape(q,d)
    #return d*grad_est.sum(axis=0)/q

def grad_coord_estimation(mu,q,x,d,kappa,target_label,const,model,orig_img,arg_targeted_attack,arg_cons):
    ### q: number of coordinates
    idx_coords_random = np.random.randint(d, size=q) ### note that ZO SCD does not rely on random direction vectors
    grad_coor_ZO = 0
    for id_coord in range(q):
        idx_coord = idx_coords_random[id_coord]
        u = np.zeros(d)
        u[idx_coord] = 1
        u = np.resize(u, x.shape)

        if arg_cons == 'uncons':
            f_old, ignore = function_evaluation_uncons(x-mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
            f_new, ignore = function_evaluation_uncons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)

        else:
            f_old, ignore = function_evaluation_cons(x-mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
            f_new, ignore = function_evaluation_cons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)

        grad_coor_ZO = grad_coor_ZO + (d / q) * (f_new - f_old) / (2 * mu) * u
    return grad_coor_ZO

def gradient_estimation_NES(mu,q,x,d,kappa,target_label,const,model,orig_img,arg_targeted_attack,arg_cons):
    # x is img_vec format in real value: w
    # m, sigma = 0, 100 # mean and standard deviation
    sigma = 100
    ## generate random direction vectors
    q_prime = int(np.ceil(q/2))
    # U_all_new = np.random.multivariate_normal(np.zeros(d), np.diag(sigma*np.ones(d) + 0), (q_prime,1))


    # if arg_cons == 'uncons':
    #     f_0=function_evaluation_uncons(x,kappa,target_label,const,model,orig_img,arg_targeted_attack)
    # else:
    #     f_0=function_evaluation_cons(x,kappa,target_label,const,model,orig_img,arg_targeted_attack)

    grad_est=0
    for i in range(q_prime):
        u = np.random.normal(0, sigma, (1,d))
        u_norm = np.linalg.norm(u)
        u = u/u_norm
        # ui = U_all_new[i, 0].reshape(-1)
        # u = ui / np.linalg.norm(ui)
        # u = np.resize(u, x.shape)
        if arg_cons == 'uncons':
            ### x+mu*u = x0 + delta + mu*u: unconstrained in R^d, constrained in [-0.5,0.5]^d
            f_tmp1, ignore = function_evaluation_uncons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
            f_tmp2, ignore = function_evaluation_uncons(x-mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
        else:
            f_tmp1, ignore = function_evaluation_cons(x+mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)
            f_tmp2, ignore = function_evaluation_cons(x-mu*u,kappa,target_label,const,model,orig_img,arg_targeted_attack)

        grad_est=grad_est+ (d/q)*u*(f_tmp1-f_tmp2)/(2*mu)
    return grad_est
    #grad_est=grad_est.reshape(q,d)
    #return d*grad_est.sum(axis=0)/q

def projection_box_eps(a_point, X, Vt, lb, up, eps = 16/256):
    ## X \in R^{d \times m}
    #d_temp = a_point.size
    #X = np.reshape(X, (X.shape[0]*X.shape[1],X.shape[2]))
    min_VtX = np.min(X, axis=0)
    max_VtX = np.max(X, axis=0)

    Lb = np.maximum(-eps, lb - min_VtX)
    Ub = np.minimum( eps, up - max_VtX)
    z_proj_temp = np.clip(a_point,Lb,Ub)
    return z_proj_temp.reshape(a_point.shape)

### projection
def projection_box(a_point, X, Vt, lb, up):
    ## X \in R^{d \times m}
    #d_temp = a_point.size
    VtX = np.sqrt(Vt)*X
    
    min_VtX = np.min(VtX, axis=0)
    max_VtX = np.max(VtX, axis=0)

    Lb = lb * np.sqrt(Vt) - min_VtX
    Ub = up * np.sqrt(Vt) - max_VtX
    
    a_temp = np.sqrt(Vt)*a_point
    z_proj_temp = np.multiply(Lb, np.less(a_temp, Lb)) + np.multiply(Ub, np.greater(a_temp, Ub)) \
                  + np.multiply(a_temp, np.multiply( np.greater_equal(a_temp, Lb), np.less_equal(a_temp, Ub)))
    #delta_proj = np.diag(1/np.diag(np.sqrt(Vt)))*z_proj_temp
    delta_proj = 1/np.sqrt(Vt)*z_proj_temp
    #print(delta_proj)
    return delta_proj.reshape(a_point.shape)

def projection_box_2(a_point, X, Vt, lb, up):
    ## X \in R^{d \times m}
    d_temp = a_point.size
    VtX = np.sqrt(Vt)@X

    min_VtX = np.min(VtX, axis=1)
    max_VtX = np.max(VtX, axis=1)

    Lb = lb * np.sqrt(Vt)@np.ones((d_temp,1)) - min_VtX.reshape((-1,1))
    Ub = up * np.sqrt(Vt)@np.ones((d_temp,1)) - max_VtX.reshape((-1,1))
    
    a_temp = np.sqrt(Vt)@(a_point.reshape((-1,1)))
    
    z_proj_temp = np.multiply(Lb, np.less(a_temp, Lb)) + np.multiply(Ub, np.greater(a_temp, Ub)) \
                  + np.multiply(a_temp, np.multiply( np.greater_equal(a_temp, Lb), np.less_equal(a_temp, Ub)))
    
    delta_proj = np.diag(1/np.diag(np.sqrt(Vt)))@z_proj_temp
    #print(delta_proj)
    return delta_proj.reshape(a_point.shape)
### replace inf or -inf in a vector to a finite value
def Inf2finite(a,val_max):
    a_temp = a.reshape(-1)
    for i_temp in range(len(a_temp)):
        test_value = a_temp[i_temp]
        if np.isinf(test_value) and test_value > 0:
            a_temp[i_temp] = val_max
        if np.isinf(test_value) and test_value < 0:
            a_temp[i_temp] = -val_max

########################################

main(args)

