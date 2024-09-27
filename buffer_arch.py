import os
from pyomo.environ import *
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import pandas as pd
import openpyxl
from datetime import datetime


def get_min_trans_mode_1(X0=1024,Y0=1024,C=256,B=64*1024, max_read_bw=10000000, max_psum_bw=32*4, min_read_bw = 0, buffer_limit=64*1024, detail=False):
    TOTAL_BUFFER_SIZE = B
    model = ConcreteModel()

    ############# value for MAC layout #################
    model.mac_rate_x           = Var(domain=PositiveIntegers, initialize=5,   bounds=(5, 8))
    model.mac_rate_y           = Var(domain=PositiveIntegers, initialize=5,   bounds=(5, 8))
    model.mac_x                = Var(domain=PositiveIntegers, initialize=128, bounds=(32, 512))
    model.mac_y                = Var(domain=PositiveIntegers, initialize=128, bounds=(32, 512))
    model.mac_c                = Var(domain=PositiveIntegers, initialize=1,   bounds=(1, 512))

    ############# constraints for MAC layout #################
    model.cons_mac_x           = Constraint(expr= model.mac_x == 2**model.mac_rate_x) ## max_x = {32, 64, 128, 512}
    model.cons_mac_y           = Constraint(expr= model.mac_y == 2**model.mac_rate_y) ## max_y = {32, 64, 128, 512}
    model.cons_mac_xyc         = Constraint(expr= model.mac_x*model.mac_y*model.mac_c == 128*128)




    ############ value for pixel/weight shape stored in the input buffer ##########
    model.x                    = Var(domain=PositiveIntegers, initialize=32) ## Sp，Sw
    model.y                    = Var(domain=PositiveIntegers, initialize=32)
    model.xc                   = Var(domain=PositiveIntegers, initialize=32)
    model.yc                   = Var(domain=PositiveIntegers, initialize=32)

    model.x_block              = Var(domain=PositiveIntegers,bounds=(1, math.ceil(X0/32)), initialize=1) # 
    model.y_block              = Var(domain=PositiveIntegers,bounds=(1, math.ceil(Y0/32)), initialize=1)
    model.xc_block             = Var(domain=PositiveIntegers,bounds=(1, C), initialize=1)
    model.yc_block             = Var(domain=PositiveIntegers,bounds=(1, C), initialize=1)

    ############ constraints for pixel/weight shape stored in the input buffer ##########
    model.cons_x               = Constraint(expr= model.x <= X0)
    model.cons_y               = Constraint(expr= model.y <= Y0)
    model.cons_x_macx          = Constraint(expr= model.x  == model.mac_x*model.x_block)
    model.cons_y_macy          = Constraint(expr= model.y  == model.mac_y*model.y_block)
    model.cons_c_macc_1        = Constraint(expr= model.xc == model.mac_c*model.xc_block)
    model.cons_c_macc_2        = Constraint(expr= model.yc == model.mac_c*model.yc_block)



    
    ############# value for share buffer (input buffer and psum buffer) ##############
    model.buf_x                = Var(domain=PositiveIntegers, initialize=1024, bounds=(0, math.ceil(X0)*32*math.ceil(C)*32)) ## needed buffer size
    model.buf_y                = Var(domain=PositiveIntegers, initialize=1024, bounds=(0, math.ceil(Y0)*32*math.ceil(C)*32))

    model.reg_rate             = Var(domain=PositiveIntegers, bounds=(1, 4)) 
    model.reg_inmac            = Var(domain=PositiveIntegers, initialize=2, bounds=(2, 16)) # {2, 4, 8, 16}
    model.buf_psum             = Var(domain=PositiveIntegers, initialize=128*1024, bounds=(128*1024, 1024*1024)) #{128K, 256K, 512K, 1024K}

    ############# constraints for share buffer (input buffer and psum buffer) ##############
    model.cons_bufx            = Constraint(expr= model.buf_x >= model.x*model.xc)
    model.cons_bufy            = Constraint(expr= model.buf_y >= model.y*model.yc)
    model.cons_bufxy           = Constraint(expr= model.buf_x + model.buf_y == TOTAL_BUFFER_SIZE-model.buf_psum)
    
    model.cons_reg             = Constraint(expr= model.reg_inmac == 2**model.reg_rate)
    model.cons_psum1           = Constraint(expr= model.buf_psum  == model.reg_inmac*128*128*4)





    ######################## value for constructing fun for min read data #########################
    model.x_rate_int           = Var(domain=PositiveIntegers, bounds=(1, math.ceil(X0/32)), initialize=math.ceil(X0/32)) # 
    model.y_rate_int           = Var(domain=PositiveIntegers, bounds=(1, math.ceil(Y0/32)), initialize=math.ceil(Y0/32))
    model.xc_rate_int          = Var(domain=PositiveIntegers, bounds=(1, math.ceil( C/32)), initialize=math.ceil( C/32)) 
    model.yc_rate_int          = Var(domain=PositiveIntegers, bounds=(1, math.ceil( C/32)), initialize=math.ceil( C/32)) 

    model.data_r_y             = Var(domain=PositiveIntegers,    initialize=Y0*C,  bounds=(Y0*C, None))
    model.data_r_x             = Var(domain=PositiveIntegers,    initialize=X0*C,  bounds=(X0*C, None))
    model.data_r_p             = Var(domain=NonNegativeIntegers, initialize=0,     bounds=(0, None))
    model.data_w_p             = Var(domain=PositiveIntegers,    initialize=X0*Y0, bounds=(X0*Y0, None))

    
    ######################## constraint for constructing fun for min read data #########################
    model.cons_x_rate_int       = Constraint(expr= model.x_rate_int  >= X0/model.x) ## 
    model.cons_y_rate_int       = Constraint(expr= model.y_rate_int  >= Y0/model.y)
    model.cons_xc_rate_int      = Constraint(expr= model.xc_rate_int >=  C/model.xc)
    model.cons_yc_rate_int      = Constraint(expr= model.yc_rate_int >=  C/model.yc)

    model.cons_x_rate_int2      = Constraint(expr= model.x_rate_int <= X0/model.x +1)
    model.cons_y_rate_int2      = Constraint(expr= model.y_rate_int <= Y0/model.y +1)
    model.cons_xc_rate_int2     = Constraint(expr= model.xc_rate_int <= C/model.xc +1)
    model.cons_yc_rate_int2     = Constraint(expr= model.yc_rate_int <= C/model.yc +1)

    model.cons_psum_1           = Constraint(expr= model.x_block*model.y_block <= (C/(C-model.xc+0.0000001))*model.reg_inmac)
    model.cons_psum_2           = Constraint(expr= model.x_block*model.y_block >= (C-model.xc)/C*model.reg_inmac)

    model.cons_datay            = Constraint(expr= (model.data_r_y >= model.x_rate_int*(Y0*C-model.buf_y)+model.buf_y))
    model.cons_detax            = Constraint(expr= (model.data_r_x >= (model.y_rate_int*(model.x*C-model.buf_x)+model.buf_x)*model.x_rate_int))





    ################ read/write bandwidth of tensor share buffer #################
    model.min_data_read         = Var(initialize=min_read_bw, bounds=(min_read_bw, None))
    model.min_data_write        = Var(initialize=4*128*128/C)

    # Data Read
    model.cons_min_data1        = Constraint(expr = model.min_data_read == (model.data_r_x + model.data_r_y)*128*128/X0/Y0/C)
    model.cons_min_data2        = Constraint(expr = model.min_data_read >= min_read_bw)
    model.cons_min_data3        = Constraint(expr = model.min_data_read <= max_read_bw)
    
    # Data Write
    model.cons_write_data       = Constraint(expr = model.min_data_write == X0*Y0*4*128*128/X0/Y0/C)

    #################  read/write bandwidth of tensor core ##################
    model.min_psum_read         = Var(initialize=128*128/C)

    model.cons_psum_write       = Constraint(expr= model.min_psum_read == 4*128*128/(model.xc/model.mac_c))
    model.cons_psum_write_max   = Constraint(expr= model.min_psum_read <= max_psum_bw)


    model.obj = Objective(expr=model.min_data_read+model.min_psum_read*1000000, sense = minimize)
    # if (DEBUG == 1):
    #     model.pprint()
    opt = SolverFactory('scip')
    solution = opt.solve(model)

    if detail:
        print(" X: %s" %    value(model.x))
        print(" Y: %s" %    value(model.y))
        print(" xc: %s" %    value(model.xc))
        print(" yc: %s" %    value(model.yc))
        print(" X rate: %s" %    value(model.x_rate_int))
        print(" Y rate: %s" %    value(model.y_rate_int))
        # print(" Z rate: %s" %    value(model.z_rate_int))
        print("Buffer for X: %s" %    value(model.buf_x/1024))
        print("Buffer for Y: %s" %    value(model.buf_y/1024))
        print("Buffer for X+Y: %s" %    value(model.buf_y/1024+model.buf_x/1024))
        print("Buffer for psum: %s" %    value(model.buf_psum/1024))
        print(value(model.data_r_x))
        print(value(model.data_r_y))
        print("Min Data Read: %s" % value(model.min_data_read))
        # print("Min Data Write: %s" % value(model.data_write_psum))
    return (
        {
        'mac layout': {'mac_x': value(model.mac_x), 'mac_y': value(model.mac_y), 'mac_c': value(model.mac_c)},
        'Buffer Manage': {'x buffer': value(model.buf_x/1024), 'y buffer': value(model.buf_y/1024), 'psum buffer': value(model.buf_psum/1024)},
        'TRAM BW': {'read bw': value(model.min_data_read), 'write bw': value(model.min_data_write)},  
        'Psum BW': value(model.min_psum_read), 
        'X shape in X_Buffer': (value(model.x), value(model.xc)), 
        'Y shape in X_Buffer':(value(model.y), value(model.yc))
        }
        )

def get_min_trans_mode_2(X0=1024,Y0=1024,C=256,B=64*1024, max_read_bw=10000000, max_psum_bw=32*4, min_read_bw = 0, buffer_limit=64*1024, detail=False):
    TOTAL_BUFFER_SIZE = B
    model = ConcreteModel()

    ############# value for MAC layout #################
    model.mac_rate_x           = Var(domain=PositiveIntegers, initialize=5,   bounds=(5, 8))
    model.mac_rate_y           = Var(domain=PositiveIntegers, initialize=5,   bounds=(5, 8))
    model.mac_x                = Var(domain=PositiveIntegers, initialize=128, bounds=(32, 512))
    model.mac_y                = Var(domain=PositiveIntegers, initialize=128, bounds=(32, 512))
    model.mac_c                = Var(domain=PositiveIntegers, initialize=1,   bounds=(1, 4))

    ############# constraints for MAC layout #################
    model.cons_mac_x           = Constraint(expr= model.mac_x == 2**model.mac_rate_x) ## max_x = {32, 64, 128, 512}
    model.cons_mac_y           = Constraint(expr= model.mac_y == 2**model.mac_rate_y) ## max_y = {32, 64, 128, 512}
    model.cons_mac_xyc         = Constraint(expr= model.mac_x*model.mac_y*model.mac_c == 128*128)




    ############ value for pixel/weight shape stored in the input buffer ##########
    model.x                    = Var(domain=PositiveIntegers, initialize=32) ## Sp，Sw
    model.y                    = Var(domain=PositiveIntegers, initialize=32)
    model.xc                   = Var(domain=PositiveIntegers, initialize=32)
    model.yc                   = Var(domain=PositiveIntegers, initialize=32)

    model.x_block              = Var(domain=PositiveIntegers,bounds=(1, math.ceil(X0/32)), initialize=1) # 
    model.y_block              = Var(domain=PositiveIntegers,bounds=(1, math.ceil(Y0/32)), initialize=1)
    model.xc_block             = Var(domain=PositiveIntegers,bounds=(1, C), initialize=1)
    model.yc_block             = Var(domain=PositiveIntegers,bounds=(1, C), initialize=1)

    ############ constraints for pixel/weight shape stored in the input buffer ##########
    model.cons_x               = Constraint(expr= model.x <= X0)
    model.cons_y               = Constraint(expr= model.y <= Y0)
    model.cons_x_macx          = Constraint(expr= model.x  == model.mac_x*model.x_block)
    model.cons_y_macy          = Constraint(expr= model.y  == model.mac_y*model.y_block)
    model.cons_c_macc_1        = Constraint(expr= model.xc == model.mac_c*model.xc_block)
    model.cons_c_macc_2        = Constraint(expr= model.yc == model.mac_c*model.yc_block)



    
    ############# value for share buffer (input buffer and psum buffer) ##############
    model.buf_x                = Var(domain=PositiveIntegers, initialize=1024, bounds=(0, math.ceil(X0)*32*math.ceil(C)*32)) ## needed buffer size
    model.buf_y                = Var(domain=PositiveIntegers, initialize=1024, bounds=(0, math.ceil(Y0)*32*math.ceil(C)*32))

    model.reg_rate             = Var(domain=PositiveIntegers, bounds=(1, 4)) 
    model.reg_inmac            = Var(domain=PositiveIntegers, initialize=2, bounds=(2, 16)) # {2, 4, 8, 16}
    model.buf_psum             = Var(domain=PositiveIntegers, initialize=128*1024, bounds=(128*1024, 1024*1024)) #{128K, 256K, 512K, 1024K}

    ############# constraints for share buffer (input buffer and psum buffer) ##############
    model.cons_bufx            = Constraint(expr= model.buf_x >= model.x*model.xc)
    model.cons_bufy            = Constraint(expr= model.buf_y >= model.y*model.yc)
    model.cons_bufxy           = Constraint(expr= model.buf_x + model.buf_y == TOTAL_BUFFER_SIZE-model.buf_psum)
    
    model.cons_reg             = Constraint(expr= model.reg_inmac == 2**model.reg_rate)
    model.cons_psum1           = Constraint(expr= model.buf_psum  == model.reg_inmac*128*128*4)





    ######################## value for constructing fun for min read data #########################
    model.x_rate_int           = Var(domain=PositiveIntegers, bounds=(1, math.ceil(X0/32)), initialize=math.ceil(X0/32)) # 
    model.y_rate_int           = Var(domain=PositiveIntegers, bounds=(1, math.ceil(Y0/32)), initialize=math.ceil(Y0/32))
    model.xc_rate_int          = Var(domain=PositiveIntegers, bounds=(1, math.ceil( C/32)), initialize=math.ceil( C/32)) 
    model.yc_rate_int          = Var(domain=PositiveIntegers, bounds=(1, math.ceil( C/32)), initialize=math.ceil( C/32)) 

    model.data_r_y             = Var(domain=PositiveIntegers,    initialize=Y0*C,  bounds=(Y0*C, None))
    model.data_r_x             = Var(domain=PositiveIntegers,    initialize=X0*C,  bounds=(X0*C, None))
    model.data_r_p             = Var(domain=NonNegativeIntegers, initialize=0,     bounds=(0, None))
    model.data_w_p             = Var(domain=PositiveIntegers,    initialize=X0*Y0, bounds=(X0*Y0, None))

    
    ######################## constraint for constructing fun for min read data #########################
    model.cons_x_rate_int       = Constraint(expr= model.x_rate_int  >= X0/model.x) ## 
    model.cons_y_rate_int       = Constraint(expr= model.y_rate_int  >= Y0/model.y)
    model.cons_xc_rate_int      = Constraint(expr= model.xc_rate_int >=  C/model.xc)
    model.cons_yc_rate_int      = Constraint(expr= model.yc_rate_int >=  C/model.yc)

    model.cons_x_rate_int2      = Constraint(expr= model.x_rate_int <= X0/model.x +1)
    model.cons_y_rate_int2      = Constraint(expr= model.y_rate_int <= Y0/model.y +1)
    model.cons_xc_rate_int2     = Constraint(expr= model.xc_rate_int <= C/model.xc +1)
    model.cons_yc_rate_int2     = Constraint(expr= model.yc_rate_int <= C/model.yc +1)

    # model.cons_psum_1           = Constraint(expr= model.x_block*model.y_block <= (C/(C-model.xc+0.0000001))*model.reg_inmac)
    # model.cons_psum_2           = Constraint(expr= model.x_block*model.y_block >= (C-model.xc)/C*model.reg_inmac)

    model.cons_detax            = Constraint(expr= (model.data_r_x == X0*C))
    model.cons_datay            = Constraint(expr= (model.data_r_y >= model.x_rate_int*(Y0*C-model.buf_y)+model.buf_y))
    model.cons_detap            = Constraint(expr= (model.data_r_p >= (model.xc_rate_int-1)*(X0*Y0-model.buf_psum)*4))




    ################ read/write bandwidth of tensor share buffer #################
    model.min_data_read         = Var(initialize=min_read_bw, bounds=(min_read_bw, None))
    model.min_data_write        = Var(initialize=4*128*128/C)

    # share buffer Data Read
    model.cons_min_data_r_1     = Constraint(expr = model.min_data_read == (model.data_r_x + model.data_r_y + model.data_r_p)*128*128/X0/Y0/C)
    model.cons_min_data_r_2     = Constraint(expr = model.min_data_read >= min_read_bw)
    model.cons_min_data_r_3     = Constraint(expr = model.min_data_read <= max_read_bw)
    
    # share buffer Data Write
    model.cons_min_write        = Constraint(expr = model.min_data_write == (model.data_r_p + X0*Y0*4)*128*128/X0/Y0/C)

    
    
    
    #################  read/write bandwidth of tensor core ##################
    model.min_psum_read         = Var(initialize=128*128/C, bounds=(128*128/C, None))
    # model.min_psum_write        = Var(bounds=(128*128/C, None))
    
    model.cons_psum_read        = Constraint(expr= model.min_psum_read == 4*128*128/(model.xc/model.mac_c))
    # model.cons_psum_read_max    = Constraint(expr= model.min_psum_read <= max_psum_bw)

    # model.cons_min_write_psum_0 = Constraint(expr = model.data_write_psum == model.x*model.y/model.xc*4)
    # model.cons_min_write_psum_1 = Constraint(expr = model.data_write_psum >= 128*128/model.xc*4)
    # model.cons_min_write_psum_2 = Constraint(expr = model.data_write_psum <= max_psum_bw)


    model.obj = Objective(expr=model.min_data_read+model.min_psum_read+model.min_data_write, sense = minimize)


    opt = SolverFactory('scip')
    solution = opt.solve(model)

    if detail:
        print(" X: %s" %    value(model.x))
        print(" Y: %s" %    value(model.y))
        print(" xc: %s" %    value(model.xc))
        print(" yc: %s" %    value(model.yc))
        print(" X rate: %s" %    value(model.x_rate_int))
        print(" Y rate: %s" %    value(model.y_rate_int))
        # print(" Z rate: %s" %    value(model.z_rate_int))
        print("Buffer for X: %s" %    value(model.buf_x/1024))
        print("Buffer for Y: %s" %    value(model.buf_y/1024))
        print("Buffer for X+Y: %s" %    value(model.buf_y/1024+model.buf_x/1024))
        print("Buffer for psum: %s" %    value(model.buf_psum/1024))
        print(value(model.data_r_x))
        print(value(model.data_r_y))
        print("Min Data Read: %s" % value(model.min_data_read))
        print("Min Data Write: %s" % value(model.min_psum_read))
    return (
        {
        'mac layout': {'mac_x': value(model.mac_x), 'mac_y': value(model.mac_y), 'mac_c': value(model.mac_c)},
        'Buffer Manage': {'x buffer': value(model.buf_x/1024), 'y buffer': value(model.buf_y/1024), 'psum buffer': value(model.buf_psum/1024)},
        'TRAM BW': {'read bw': value(model.min_data_read), 'write bw': value(model.min_data_write)},  
        'Psum BW': value(model.min_psum_read), 
        'X shape in X_Buffer': (value(model.x), value(model.xc)), 
        'Y shape in X_Buffer':(value(model.y), value(model.yc))
        }
        )


def get_arch_info(P=256, W=512, C=1024, Buffer = 1024, max_read_bw=512, max_psum_bw=512, mode='psum-pixel-weight', detail=False): ## Buffer: KB
    # mode = {
    #   psum-pixel-weight, 
    #   psum-weight-pixel, 
    #   pixel-psum-weight, 
    #   weight-psum-pixel
    # }

    min_bw = (P*C+W*C)/((P*W*C)/128/128)
    Arch_param = ()

    mode_list = mode.split('-')
    if (mode_list[0] == 'psum'):
        mode_ = 'Psum'
        if (mode_list[1] == 'pixel'):
            X = P
            Y = W
        else:
            X = W
            Y = P
    elif(mode_list[0] == 'pixel'):
        X = P
        Y = W
    else:
        X = W
        Y = P

    # Arch_param:
    #   'mac layout': {
    #       'mac_x': 128, 
    #       'mac_y': 128, 
    #       'mac_c': 1
    #   },
    #   'Buffer Manage': {
    #       'x buffer': 32 KB, 
    #       'y buffer': 32 KB, 
    #       'psum buffer': 256 KB 
    #   }, 
    #   'TRAM BW': {
    #       'read bw': , 
    #       'write bw': 
    #   }  
    #   'Psum BW': , 
    #
    #   'X shape in X_Buffer': (value(model.x), value(model.xc)), 
    #   'Y shape in Y_Buffer':(value(model.y), value(model.yc))


    if (mode_ == 'psum'):
        Arch_param = get_min_trans_mode_1(X0=X,Y0=Y,C=C,B=Buffer*1024, min_read_bw = min_bw, max_read_bw=max_read_bw, max_psum_bw=max_psum_bw, detail=detail)
    else:
        Arch_param = get_min_trans_mode_2(X0=X,Y0=Y,C=C,B=Buffer*1024, min_read_bw = min_bw, max_read_bw=max_read_bw, max_psum_bw=max_psum_bw, detail=detail)
    return(Arch_param)# print(Arch_param)
# print(min_bw)
Arch_param = get_arch_info()
print(Arch_param)
