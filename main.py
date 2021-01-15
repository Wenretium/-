import math
from sklearn.utils import resample
import random
data3 = [6, 5, 16, 7, 3, 14, 16, 13, 3, 7, 1, 15, 12, 6, 16, 3, 14]
data3.sort()
print(data3)

# 计算信息熵
def Ent(D):
    pos, neg = cal_pos_neg(D)
    if pos != 0 and neg != 0:
        return -(pos/len(D)*math.log(pos/len(D), 2)+neg/len(D)*math.log(neg/len(D), 2))
    elif pos == 0 and neg != 0:
        return -(0+neg/len(D)*math.log(neg/len(D), 2))
    elif pos != 0 and neg == 0:
        return -(pos/len(D)*math.log(pos/len(D), 2) + 0)
    else:
        return 0

# 统计正反例数量
def cal_pos_neg(D):
    pos = 0
    neg = 0
    for i in D:
        if i['好瓜'] == '是':
            pos += 1
        else:
            neg += 1
    return pos, neg

# 根据属性及对应的取值划分数据集 D
def split_att(D, att):
    att_value = att_value_list[att]
    if len(att_value) == 3:
        D1 = []
        D2 = []
        D3 = []
        for melon in D:
            if melon[att] == att_value[0]:
                D1.append(melon)
            elif melon[att] == att_value[1]:
                D2.append(melon)
            else:
                D3.append(melon)
        return D1, D2, D3
    elif len(att_value) == 2:
        D1 = []
        D2 = []
        for melon in D:
            if melon[att] == att_value[0]:
                D1.append(melon)
            else:
                D2.append(melon)
        return D1, D2

# 计算信息增益
def Gain(D, att):
    D_spilt = split_att(D, att)
    sum = 0
    for d in D_spilt:
        sum += len(d)/len(D)*Ent(d)
    return Ent(D)-sum

# 计算增益率
def Gain_ratio(D, att):
    D_split = split_att(D, att)
    IV_a = 0
    D_l = len(D)
    for d in D_split:
        if len(d) == 0:
            IV_a += 0
        else:
            IV_a += -len(d)/D_l*math.log(len(d)/D_l, 2)
    if IV_a == 0:  # 取为无穷接近于0的数：1e-10
        IV_a = 1e-10
    print(att,Gain(D, att),IV_a)
    return Gain(D, att)/IV_a

# 计算基尼值
def Gini(D):
    pos, neg = cal_pos_neg(D)
    D_l = len(D)
    if D_l != 0:
        return 1-(pos/D_l)**2-(neg/D_l)**2
    else:
        return 1

# 计算基尼系数
def Gini_index(D, att):
    D_split = split_att(D, att)
    D_l = len(D)
    Gini_index_a = 0
    for d in D_split:
        Gini_index_a += len(d)/D_l*Gini(d)
    return Gini_index_a

# 检查 D 内元素是否属于一个类别
def check_type(D):
    pos, neg = cal_pos_neg(D)
    if pos*neg == 0:  # 其中一个为 0
        return True
    else:
         return False

def Decision_Tree(data, att_list, layer_num, score_function):
    # 控制属性选择范围
    k = int(math.log(len(att_list), 2))+1  # 向上取整
    # print(att_list,k)
    att_list = random.sample(att_list, k)  # 不放回抽样 k 个
    # 特殊情况，结束递归
    # 决策树已达到规定层数
    if layer_num >= 2:
        tpos, tneg = cal_pos_neg(data)
        print('-标记为', end=' ')
        if tpos > tneg:
            print('是', end=' ')
        else:
            print('否', end=' ')
        print('类叶结点')
        return
    # print(att_list)
    # 集合中元素全为同一类型，不用继续划分
    if check_type(data):
        print('-标记为', data[0]['好瓜'], '类叶结点')
        return
    # 用来划分的属性集为空
    elif att_list == []:
        tpos, tneg = cal_pos_neg(data)
        print('-标记为',end=' ')
        if tpos>tneg:
            print('是', end=' ')
        else:
            print('否', end=' ')
        print('类叶结点')
        return

    # 选出最优划分属性
    # 对每个属性逐个计算
    score_list = {}
    for att in att_list:
        score_list[att] = score_function(data, att)
    print('---score---')
    for i in score_list:
        print(i, round(score_list[i], 4))
    # 选出信息增益最大者
    if score_function == Gini_index:
        chosen_att = min(score_list, key=score_list.get)
    else:
        chosen_att = max(score_list, key=score_list.get)

    print('第', layer_num, '层, 选取', chosen_att, '作为划分依据')
    # 对子数据集递归划分
    for i, melon_data_pre in enumerate(split_att(data, chosen_att)):
        print('进入值',att_value_list[chosen_att][i])
        if melon_data_pre == []:
            # print(melon_data_pre)
            print('标记为', data[0]['好瓜'], '类叶结点')
        else:
            # print(layer_num+1, melon_data_pre)
            att_list_c = att_list.copy()
            att_list_c.remove(chosen_att)
            Decision_Tree(melon_data_pre, att_list_c, layer_num+1, score_function)
        print('----------返回父节点')


melon_data = [{'色泽':'青绿', '根蒂':'蜷缩','敲声':'浊响','纹理':'清晰','脐部':'凹陷','触感':'硬滑','好瓜':'是'},
        {'色泽':'乌黑', '根蒂':'蜷缩','敲声':'沉闷','纹理':'清晰','脐部':'凹陷','触感':'硬滑','好瓜':'是'},
        {'色泽':'乌黑', '根蒂':'蜷缩','敲声':'浊响','纹理':'清晰','脐部':'凹陷','触感':'硬滑','好瓜':'是'},
        {'色泽':'青绿', '根蒂':'蜷缩','敲声':'沉闷','纹理':'清晰','脐部':'凹陷','触感':'硬滑','好瓜':'是'},
        {'色泽':'浅白', '根蒂':'蜷缩','敲声':'浊响','纹理':'清晰','脐部':'凹陷','触感':'硬滑','好瓜':'是'},
        {'色泽':'青绿', '根蒂':'稍蜷','敲声':'浊响','纹理':'清晰','脐部':'稍凹','触感':'软粘','好瓜':'是'},
        {'色泽':'乌黑', '根蒂':'稍蜷','敲声':'浊响','纹理':'稍糊','脐部':'稍凹','触感':'软粘','好瓜':'是'},
        {'色泽':'乌黑', '根蒂':'稍蜷','敲声':'浊响','纹理':'清晰','脐部':'稍凹','触感':'硬滑','好瓜':'是'},
        {'色泽':'乌黑', '根蒂':'稍蜷','敲声':'沉闷','纹理':'稍糊','脐部':'稍凹','触感':'硬滑','好瓜':'否'},
        {'色泽':'青绿', '根蒂':'硬挺','敲声':'清脆','纹理':'清晰','脐部':'平坦','触感':'软粘','好瓜':'否'},
        {'色泽':'浅白', '根蒂':'硬挺','敲声':'清脆','纹理':'模糊','脐部':'平坦','触感':'硬滑','好瓜':'否'},
        {'色泽':'浅白', '根蒂':'蜷缩','敲声':'浊响','纹理':'模糊','脐部':'平坦','触感':'软粘','好瓜':'否'},
        {'色泽':'青绿', '根蒂':'稍蜷','敲声':'浊响','纹理':'稍糊','脐部':'凹陷','触感':'硬滑','好瓜':'否'},
        {'色泽':'浅白', '根蒂':'稍蜷','敲声':'沉闷','纹理':'稍糊','脐部':'凹陷','触感':'硬滑','好瓜':'否'},
        {'色泽':'乌黑', '根蒂':'稍蜷','敲声':'浊响','纹理':'清晰','脐部':'稍凹','触感':'软粘','好瓜':'否'},
        {'色泽':'浅白', '根蒂':'蜷缩','敲声':'浊响','纹理':'模糊','脐部':'平坦','触感':'硬滑','好瓜':'否'},
        {'色泽':'青绿', '根蒂':'蜷缩','敲声':'沉闷','纹理':'稍糊','脐部':'稍凹','触感':'硬滑','好瓜':'否'}]
att_list = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
att_value_list = {'色泽':['青绿','乌黑','浅白'],
            '根蒂':['蜷缩','稍蜷','硬挺'],
            '敲声':['浊响','沉闷','清脆'],
            '纹理':['清晰','稍糊','模糊'],
            '脐部':['凹陷','稍凹','平坦'],
            '触感':['硬滑','软粘']}
data1 = [10, 6, 14, 16, 4, 6, 11, 4, 15, 4, 4, 2, 2, 13, 4, 9, 14]
data2 = [3, 4, 13, 14, 9, 7, 14, 9, 4, 3, 2, 7, 5, 13, 12, 2, 1]
data3 = [6, 5, 16, 7, 3, 14, 16, 13, 3, 7, 1, 15, 12, 6, 16, 3, 14]
melon_data_pre = []
for i in data3:
    melon_data_pre.append(melon_data[i-1])
# print(len(melon_data_pre))

Decision_Tree(melon_data_pre, att_list, 0, Gini_index)
