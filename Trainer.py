Import Network
import h5py
import numpy as np
import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn 
import paddle.nn.functional as F
import paddle.vision.transforms as TF
import random
import cv2

import matplotlib.pyplot as plt
%matplotlib inline
import time

model_path = './output/'

reader = data_reader(cfg)

def adaIN_trainer(x1, x2, munit_1, munit_2, munit_1_optimizer, munit_2_optimizer, alpha=1):

    c1 = munit_2.content_encode(x1)
    s2 = munit_2.style_code  
    t_2 = adain(s2, c1)
    t_2 = alpha * t_2 + (1 - alpha) * c1
    g_t2 = munit_2.decoder(t_2)
    gc_1 = munit_2.content_encode(g_t2)
    gs_2 = munit_2.style_encode(g_t2)

    c2 = munit_1.content_encode(x2)
    s1 = munit_1.style_code  
    t_1 = adain(s1, c2)
    t_1 = alpha * t_1 + (1 - alpha) * c2
    g_t1 = munit_1.decoder(t_1)
    gc_2 = munit_1.content_encode(g_t1)
    gs_1 = munit_1.style_encode(g_t1)

    loss_gen_1 = munit_1.disc.calc_gen_loss(g_t1)
    loss_gen_2 = munit_2.disc.calc_gen_loss(g_t2)
    loss_rec_x_1 = munit_1.calc_l1_loss(g_t1, x2)
    loss_rec_x_2 = munit_2.calc_l1_loss(g_t2, x1)
    loss_rec_c_1 = munit_2.calc_l1_loss(gc_1, c1)
    loss_rec_c_2 = munit_1.calc_l1_loss(gc_2, c2)
    loss_rec_s_1 = munit_1.calc_l1_loss(gs_1, s1)
    loss_rec_s_2 = munit_2.calc_l1_loss(gs_2, s2)

    t_121 = adain(s1, gc_1)
    x_121 = munit_1.decoder(t_121)
    t_212 = adain(s2, gc_2)
    x_212 = munit_2.decoder(t_212)
    loss_rec_cycle_1=munit_1.calc_l1_loss(x1, x_121)
    loss_rec_cycle_2=munit_2.calc_l1_loss(x2, x_212)

    loss = 1*(loss_gen_1+loss_gen_2)+\
        10*(loss_rec_x_1+loss_rec_x_2)+\
        1*(loss_rec_c_1+loss_rec_c_2)+\
        1*(loss_rec_s_1+loss_rec_s_2)+\
        10*(loss_rec_cycle_1+loss_rec_cycle_2)

    munit_1_optimizer.clear_grad()
    munit_2_optimizer.clear_grad()
    loss.backward()
    munit_1_optimizer.minimize(loss)
    munit_2_optimizer.minimize(loss)

    return loss.numpy()
