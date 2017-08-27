#!usr/bin/env python2
# -*- coding:utf-8 -*-


from math import log
from PIL import Image, ImageDraw
import zlib

my_data = [['slashdot', 'USA', 'yes', 18, 'None'],
           ['google', 'France', 'yes', 23, 'Premium'],
           ['digg', 'USA', 'yes', 24, 'Basic'],
           ['kiwitobes', 'France', 'yes', 23, 'Basic'],
           ['google', 'UK', 'no', 21, 'Premium'],
           ['(direct)', 'New Zealand', 'no', 12, 'None'],
           ['(direct)', 'UK', 'no', 21, 'Basic'],
           ['google', 'USA', 'no', 24, 'Premium'],
           ['slashdot', 'France', 'yes', 19, 'None'],
           ['digg', 'USA', 'no', 18, 'None'],
           ['google', 'UK', 'no', 18, 'None'],
           ['kiwitobes', 'UK', 'no', 19, 'None'],
           ['digg', 'New Zealand', 'yes', 12, 'Basic'],
           ['slashdot', 'UK', 'no', 21, 'None'],
           ['google', 'UK', 'yes', 18, 'Basic'],
           ['kiwitobes', 'France', 'yes', 19, 'Basic']]


# 创建决策节点
class decidenode():
    def __init__(self, col=-1, value=None, result=None, tb=None, fb=None):
        self.col = col  # 待检验的判断条件所对应的列索引值
        self.value = value  # 为了使结果为true，当前列要匹配的值
        self.result = result  # 叶子节点的值
        self.tb = tb  # true下的节点
        self.fb = fb  # false下的节点


# 对数值型和离散型数据进行分类
def DivideSet(rows, column, value):
    splitfunction = None
    if isinstance(value, int) or isinstance(value, float):
        splitfunction = lambda x: x >= value
    else:
        splitfunction = lambda x: x == value

    set1 = [row for row in rows if splitfunction(row[column])]
    set2 = [row for row in rows if not splitfunction(row[column])]
    return (set1, set2)


# 计算数据所包含的实例个数
def UniqueCount(rows):
    result = {}
    for row in rows:
        r = row[len(row) - 1]
        result.setdefault(r, 0)
        result[r] += 1
    return result


# 计算Gini impurity
def GiniImpurity(rows):
    total = len(rows)
    counts = uniquecounts(rows)
    imp = 0
    for k1 in counts:
        p1 = float(counts[k1]) / total
        for k2 in counts:
            if k1 == k2: continue
            p2 = float(counts[k2]) / total
            imp += p1 * p2
    return imp


# 计算信息熵Entropy
def entropy(rows):
    log2 = lambda x: log(x) / log(2)
    results = UniqueCount(rows)
    # Now calculate the entropy
    ent = 0.0
    for r in results.keys():
        p = float(results[r]) / len(rows)
        ent = ent - p * log2(p)
    return ent


# 计算方差(当输出为连续型的时候，用方差来判断分类的好或坏，决策树两边分别是比较大的数和比较小的数)
# 可以通过后修剪来合并叶子节点
def variance(rows):
    if len(rows) == 0: return 0
    data = [row[len(rows) - 1] for row in rows]
    mean = sum(data) / len(data)
    variance = sum([(d - mean) ** 2 for d in data]) / len(data)
    return variance


###############################################################33
# 创建决策树递归
def BuildTree(rows, judge=entropy):
    if len(rows) == 0: return decidenode()

    # 初始化值
    best_gain = 0
    best_value = None
    best_sets = None
    best_col = None
    S = judge(rows)

    # 获得最好的gain
    for col in range(len(rows[0]) - 1):
        total_value = {}
        for row in rows:
            total_value[row[col]] = 1
        for value in total_value.keys():
            (set1, set2) = DivideSet(rows, col, value)

            # 计算信息增益，将最好的保存下来
            s1 = float(len(set1)) / len(rows)
            s2 = float(len(set2)) / len(rows)
            gain = S - s1 * judge(set1) - s2 * judge(set2)
            if gain > best_gain:
                best_gain = gain
                best_value = value
                best_col = col
                best_sets = (set1, set2)
                # 创建节点
    if best_gain > 0:
        truebranch = BuildTree(best_sets[0])
        falsebranch = BuildTree(best_sets[1])
        return decidenode(col=best_col, value=best_value, tb=truebranch, fb=falsebranch)
    else:
        return decidenode(result=UniqueCount(rows))


        # 打印文本形式的tree


def PrintTree(tree, indent=''):
    if tree.result != None:
        print str(tree.result)
    else:
        print '%s:%s?' % (tree.col, tree.value)
        print indent, 'T->',
        PrintTree(tree.tb, indent + '  ')
        print indent, 'F->',
        PrintTree(tree.fb, indent + '  ')


def getwidth(tree):
    if tree.tb == None and tree.fb == None: return 1
    return getwidth(tree.tb) + getwidth(tree.fb)


def getdepth(tree):
    if tree.tb == None and tree.fb == None: return 0
    return max(getdepth(tree.tb), getdepth(tree.fb)) + 1


# 打印图表形式的tree
def drawtree(tree, jpeg='tree.jpg'):
    w = getwidth(tree) * 100
    h = getdepth(tree) * 100 + 120
    img = Image.new('RGB', (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    drawnode(draw, tree, w / 2, 20)
    img.save(jpeg, 'JPEG')


def drawnode(draw, tree, x, y):
    if tree.result == None:
        # Get the width of each branch
        w1 = getwidth(tree.fb) * 100
        w2 = getwidth(tree.tb) * 100
        # Determine the total space required by this node
        left = x - (w1 + w2) / 2
        right = x + (w1 + w2) / 2
        # Draw the condition string
        draw.text((x - 20, y - 10), str(tree.col) + ':' + str(tree.value), (0, 0, 0))
        # Draw links to the branches
        draw.line((x, y, left + w1 / 2, y + 100), fill=(255, 0, 0))
        draw.line((x, y, right - w2 / 2, y + 100), fill=(255, 0, 0))
        # Draw the branch nodes
        drawnode(draw, tree.fb, left + w1 / 2, y + 100)
        drawnode(draw, tree.tb, right - w2 / 2, y + 100)
    else:
        txt = ' \n'.join(['%s:%d' % v for v in tree.result.items()])
        draw.text((x - 20, y), txt, (0, 0, 0))



        # 对新实例进行查询


def classify(observation, tree):
    if tree.result != None:
        return tree.result
    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == value:
                branch = tree.tb
            else:
                branch = tree.fb
        return classify(observation, branch)


        # 后剪枝,设定一个阈值mingain来后剪枝，当合并后熵增加的值小于原来的值，就合并


def prune(tree, mingain):
    if tree.tb.result == None:
        prune(tree.tb, mingain)
    if tree.fb.result == None:
        prune(tree.fb, mingain)

    if tree.tb.result != None and tree.fb.result != None:
        tb1, fb1 = [], []
        for v, c in tree.tb.result.items():
            tb1 += [[v]] * c  # 这里是为了跟row保持一样的格式，因为UniqueCount就是对这种进行的计算

        for v, c in tree.fb.result.items():
            fb1 += [[v]] * c

        delta = entropy(tb1 + fb1) - (entropy(tb1) + entropy(fb1) / 2)
        if delta < mingain:
            tree.tb, tree.fb = None, None
            tree.result = UniqueCount(tb1 + fb1)

            # 对缺失属性的数据进行查询


def mdclassify(observation, tree):
    if tree.result != None:
        return tree.result

    if observation[tree.col] == None:
        tb, fb = mdclassify(observation, tree.tb), mdclassify(observation, tree.fb)  # 这里的tr跟fr实际是这个函数返回的字典
        tbcount = sum(tb.values())
        fbcount = sum(fb.values())
        tw = float(tbcount) / (tbcount + fbcount)
        fw = float(fbcount) / (tbcount + fbcount)

        result = {}
        for k, v in tb.items():
            result.setdefault(k, 0)
            result[k] = v * tw
        for k, v in fb.items():
            result.setdefault(k, 0)
            result[k] = v * fw
        return result

    else:
        v = observation[tree.col]
        branch = None
        if isinstance(v, int) or isinstance(v, float):
            if v >= tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        else:
            if v == tree.value:
                branch = tree.tb
            else:
                branch = tree.fb
        return mdclassify(observation, branch)


def main():  # 以下内容为我测试决策树的代码
    a = BuildTree(my_data, 0.01)
    print a

    PrintTree(a)
    #   drawtree(a,jpeg='treeview.jpg')
    prune(a, 0.1)
    PrintTree(a)
    prune(a, 1)
    PrintTree(a)

    mdclassify(['google', 'France', None, None], a)
    print mdclassify(['google', 'France', None, None], a)

    mdclassify(['google', None, 'yes', None], a)
    print mdclassify(['google', None, 'yes', None], a)


if __name__ == '__main__':
    main()