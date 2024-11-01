# SPDX-License-Identifier: MPL-2.0
#
# ODAT-SE -- an open framework for data analysis
# Copyright (C) 2020- The University of Tokyo
#
# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0.
# If a copy of the MPL was not distributed with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import toml as tm
import sys
import datetime
import os
import math as m
import collections as cl
import argparse
import decimal

def read_toml():
    #input.tomlを取得
    dict_toml= tm.load(inputfile)

    #変数の範囲の最大値・最小値とTがlogスケールかを取得
    x1_min,x2_min= dict_toml['algorithm']['param']['min_list']
    x1_max,x2_max= dict_toml['algorithm']['param']['max_list']
    T_min= dict_toml['algorithm']['exchange']['Tmin']
    T_max= dict_toml['algorithm']['exchange']['Tmax']
    Tlog= True #dict_toml['algorithm']['exchange']['Tlogspace']
    #Tlogはブーリアン型

    #結果を出力したディレクトリ名を取得
    output_dicname= dict_toml['base']['output_dir']

    #探索回数を取得
    numsteps= int(dict_toml['algorithm']['exchange']['numsteps'])
    numsteps_exchange= int(dict_toml['algorithm']['exchange']['numsteps_exchange'])
    
    name_for_simulate = ''
    return x1_min, x2_min, x1_max, x2_max, T_max, T_min, Tlog, output_dicname, numsteps, numsteps_exchange, name_for_simulate


def read_result():

    #各ディレクトリから得た値を格納する変数
    x1= []
    x2= []
    fx= []
    T= []
    step= range(1,numsteps+1,1)

    #各探索プロセッサにおけるfxの最大値・最小値を格納する変数
    fx_max_list= []
    fx_min_list= []

    #以下よりファイルの読み込み
    i= 0
    while i <= number_of_replica-1 :
        print('{0}/{1} ファイル読み込み開始'.format(i+1,number_of_replica))
        #リストの中にi番目のディレクトリの値を格納するリストを作成
        x1.append([])
        x2.append([])
        fx.append([])
        T.append([])

        #ファイルを開き、改行区切りのリストにする。
        f=open('{0}/{1}/result.txt'.format(output_dicname,i))
        frs= f.read().splitlines()
        f.close()

        #繰り返し処理で値を抽出、float型へ変換
        #jが1からスタートなのはresult.txtの1行目には結果の値が入っていないため
        j= 1
        while j< len(frs):
            sp= frs[j].split()
            x1[i].append(float(sp[-2]))
            x2[i].append(float(sp[-1]))
            fx[i].append(float(sp[-3]))
            T[i].append(float(sp[-4]))
            j+= 1
        
        #i番目のresult.txtにおけるfxの最大値・最小値を格納
        fx_max_list.append(max(fx[i]))
        fx_min_list.append(min(fx[i]))
        i+= 1

    print('読み込み完了')

    #読み込みができてるかの確認
    #print(x1[0][0:5],'\n\n',x1[1][0:5],'\n\n',x1[2][0:5],'\n\n',x1[3][0:5])
    return x1, x2, fx, T, step, max(fx_max_list), min(fx_min_list)   


def sort_T():
    sorted_x1= []
    sorted_x2= []
    sorted_fx= []
    replica_num= []
    #original_x1= cp.deepcopy(x1)
    #original_x2= cp.deepcopy(x2)
    #original_fx= cp.deepcopy(fx)
    
    Tvalue= make_Tvalue_list()

    i= 0
    while i < len(Tvalue):
        sorted_x1.append([])
        sorted_x2.append([])
        sorted_fx.append([])
        replica_num.append([])
        i+= 1
    
    i = int(numsteps*burn_in)
    while i < numsteps:
        j= 0
        while j < number_of_replica:
            k= 0
            while k < len(Tvalue):
                if Tvalue[k] == T[j][i]:
                    sorted_x1[k].append(x1[j][i])   
                    sorted_x2[k].append(x2[j][i])
                    sorted_fx[k].append(fx[j][i])
                    replica_num[k].append(j)
                k+= 1
            j+= 1
        i+= 1

    sorted_T= True
    return sorted_x1, sorted_x2, sorted_fx, replica_num, Tvalue, sorted_T


def make_Tvalue_list():
    Tvalue= []
    i= 0
    while i < number_of_replica :
        Tvalue.extend(list(cl.Counter(T[i]).keys()))
        i+= 1
    Tvalue= list(cl.Counter(Tvalue).keys())
    return Tvalue

def soloplot():
    #繰り返し処理でプロット・図の保存
    print('グラフ描画開始')
    min_l = []
    max_l = []
    
    l= 0
    while l< number_of_replica:
        fig = plt.figure(figsize= (8,8))
        ax = fig.add_subplot()
    
        num_of_sample = len(sorted_x1[l][:])
        weight_l = np.ones(num_of_sample)/num_of_sample

        hst2d = ax.hist2d(
                sorted_x1[l][:], sorted_x2[l][:],
                norm= clr.LogNorm(vmin=10**-4, vmax=10**-1),
                range=[ [x1_min,x1_max] , [x2_min,x2_max] ],
                cmap= 'Reds', weights=weight_l , bins=100)

        sum_allspace = np.sum(hst2d[0])
        max_in_one_space = hst2d[0].max()
        min_in_one_space = hst2d[0][np.nonzero(hst2d[0])].min()
        min_l.append(min_in_one_space)
        max_l.append(max_in_one_space)
        
        cb = fig.colorbar(hst2d[3])
        
        temp_for_title = decimal.Decimal(1/Tvalue[l]).quantize(
                decimal.Decimal('0.00001'), rounding=decimal.ROUND_HALF_UP
                    )
        figtitlestr = f'beta = {temp_for_title}'

        ax.set_title(figtitlestr)
        
        line_x1 = np.arange(x1_min,x1_max+1,1)
        line_x2_1 = line_x1
        line_x2_2 = -line_x1 + 1
        ax.plot(line_x1,line_x2_1,c="black",alpha=0.3,lw=0.5)
        ax.plot(line_x1,line_x2_2,c="black",alpha=0.3,lw=0.5)
        #'''
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(x1_min,x1_max)
        ax.set_ylim(x2_min,x2_max)
        ax.set_aspect('equal')
        #軸サイズ設定と図の保存
        fig.savefig(f'{dirname}/{l}_beta_{temp_for_title}_burn_in_{burn_in}.png',dpi=300)             
        l+= 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--mpiprocess',help='MPI process')
    parser.add_argument('-i','--inputfile',help='input toml file')
    parser.add_argument('-b','--burn_in',help='burn-in ratio')
    args = parser.parse_args()
    
    number_of_replica = int(args.mpiprocess)
    inputfile = args.inputfile
    burn_in = float(args.burn_in)
    
    x1_min, x2_min, x1_max, x2_max, T_max, T_min, Tlog, output_dicname, numsteps, numsteps_exchange, name_for_simulate= read_toml()
    
    #result.txtから読み取り
    x1, x2, fx, T, step, fx_max, fx_min = read_result()
    
    #グラフタイトル・ファイル名用に実行した時間の年日時間分を取得
    time= datetime.date.today().strftime("%Y%m%d")

    sorted_x1, sorted_x2, sorted_fx, replica_num, Tvalue, sorted_T= sort_T()
    
    dirname= '{0}_histogram'.format(time)
    if os.path.exists(dirname) == False:
        os.makedirs(dirname)
 
    soloplot()
