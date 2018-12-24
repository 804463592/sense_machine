import copy
from matplotlib import pyplot as plt
from matplotlib import animation
training_set = [[(1, 2), 1], [(2, 3), 1], [(3, 1), -1], [(4, 2), -1],[(4, 5), -1],[(2, 1), 1],[(5, 2), -1],[(3, 3), -1]] # 训练数据集
w = [0, 0]    # 参数初始化
b = 0
history = []  # 用来记录每次更新过后的w,b


def update(item):
      """ 随机梯度下降更新参数
       :param item: 参数是分类错误的点
       :return: nothing 无返回值
       """
      global w, b, history # 把w, b, history声明为全局变量
      w[0] += 1 * item[1] * item[0][0] # 根据误分类点更新参数,这里学习效率设为1
      w[1] += 1 * item[1] * item[0][1]
      b += 1 * item[1]
      history.append([copy.copy(w), b]) # 将每次更新过后的w,b记录在history数组中


def cal(item):

          """
          计算item到超平面的距离,输出yi(w*xi+b)
          （我们要根据这个结果来判断一个点是否被分类错了。如果yi(w*xi+b)>0,则分类错了）
          :param item:
          :return:
          """
          res = 0
          for i in range(len(item[0])): # 迭代item的每个坐标，对于本文数据则有两个坐标x1和x2
              res += item[0][i] * w[i]
          res += b
          res *= item[1] # 这里是乘以公式中的yi
          return res



def check():
       """
          检查超平面是否已将样本正确分类
          :return: true如果已正确分类则返回True
          """
       flag = False
       for item in training_set:
           if cal(item) <= 0: # 如果有分类错误的
              flag = True # 将flag设为True
              update(item) # 用误分类点更新参数
       if not flag: # 如果没有分类错误的点了
          print("最终结果: w: " + str(w) + "b: " + str(b)) # 输出达到正确结果时参数的值
       return flag # 如果 已正确分类则返回True,否则返回False


if __name__ == "__main__":
   for i in range(100): # 迭代100遍
       if not check():
          break # 如果已正确分类，则结束迭代
   print("参数w,b更新过程：", history)  # 可视化过于麻烦，暂时不做


