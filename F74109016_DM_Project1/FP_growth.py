from collections import defaultdict
from itertools import chain, combinations
class Node:
    def __init__(self,itemName,support,parentNode):
        self.itemName = itemName # 將item放在這裡
        self.support = support
        self.parent  = parentNode
        self.children = {}
        self.next = None 
    def increment(self, support): self.support += support
    
def changeToList(transactionList): # 將資料轉成list 想要的格式而已
    map ,countitem = {},{}
    for data in transactionList:
        if data[0] not in map       : map[data[0]] = []
        if data[2] not in countitem : countitem[data[2]] = 0
        countitem[data[2]]  += 1  
        map[data[0]].append(data[2])
    input_data = []
    for key, value in map.items(): input_data.append(value)
    itemSetList = [set(transaction) for transaction in input_data]
    supportlist = [1] * len(input_data)
    return itemSetList, supportlist 

def CreateTree(itemSetList,supportlist,min_sup):
    HeaderTable = defaultdict(int)
    for idx, itemSet in enumerate(itemSetList):
        for item in itemSet:  HeaderTable[item] += supportlist[idx]
    # 先過濾第一層少於SUPPORT的item
    HeaderTable = dict((item, sup) for item, sup in HeaderTable.items() if sup >= min_sup)
    if(len(HeaderTable) == 0): return None, None
    # 將HeaderTable轉換成list 且每個HeaderTable存放出現的item在樹中的位置的list
    for item in HeaderTable: HeaderTable[item] = [HeaderTable[item], None]
    
    FPTreeHeader = Node('Header',1,None) # 樹的頭
    # 由大到小排序，因為這樣建立的樹才不會頭部是出現次數太少的，
    # 如果頭部是出現次數太少的，會導致樹的分支太多，就沒有共用prefix path的效果了
    for idx, itemSet in enumerate(itemSetList):
        itemSet = [item for item in itemSet if item in HeaderTable]
        itemSet.sort(key=lambda item: HeaderTable[item][0], reverse=True)
        currentNode = FPTreeHeader
        for item in itemSet: currentNode = UpdateTree(item, currentNode, HeaderTable, supportlist[idx])
    return FPTreeHeader, HeaderTable

def UpdateHeaderTable(item, targetNode, HeaderTable):
    if(HeaderTable[item][1] == None):
        HeaderTable[item][1] = targetNode
    else:
        currentNode = HeaderTable[item][1]
        # 將targetNode加入HeaderTable
        while currentNode.next != None: currentNode = currentNode.next
        currentNode.next = targetNode 
def UpdateTree(item,treeNode,HeaderTable,support):
    if item in treeNode.children: treeNode.children[item].increment(support) # 將小孩每個都加上support
    else:
        newItemNode = Node(item,support,treeNode)
        treeNode.children[item] = newItemNode
        # 更新HeaderTable將新節點加入linked list
        UpdateHeaderTable(item, newItemNode, HeaderTable)
    return treeNode.children[item]

def AscendTree(node, prefixPath):
    if node.parent != None:
        prefixPath.append(node.itemName)
        AscendTree(node.parent, prefixPath)
def findPrefixPath(basePat, HeaderTable):
    # First node in linked list
    treeNode = HeaderTable[basePat][1]
    condionalPatterns = []
    supportlist = []
    while treeNode != None:
        prefixPath = []
        AscendTree(treeNode, prefixPath)
        if(len(prefixPath) > 1):
            # 存放prefixPath與對應的support
            condionalPatterns.append(prefixPath[1:])
            supportlist.append(treeNode.support)
        treeNode = treeNode.next
    return condionalPatterns, supportlist

def FindFrequentItemSet(HeaderTable, minSup, preFix, freqItemList):
    # Sort the items with support and create a list
    sortedItemList = [item[0] for item in sorted(list(HeaderTable.items()), key=lambda p:p[1][0])] 
    # Start with the lowest support item because items are sorted in increasing support
    for item in sortedItemList:  
        # Pattern growth 將 suffix pattern 與 
        # frequent patterns 結合產生 conditional FP-tree
        NewFrequentSet = preFix.copy()
        NewFrequentSet.add(item)
        support = HeaderTable[item][0]
        freqItemList[frozenset(NewFrequentSet)] = support
        conditionalPattBase, supportlist = findPrefixPath(item, HeaderTable) 
        # 用 conditional pattern base 建樹
        conditionalTree, NewHeaderTable = CreateTree(conditionalPattBase, supportlist, minSup) 
        if NewHeaderTable != None: FindFrequentItemSet(NewHeaderTable, minSup, NewFrequentSet, freqItemList)  # 遞迴找frequent ItemSet
          

def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

def FindAssociationRule(FrequentItemSet, minConf,total):
    AssociationRule = []
    for itemSet in FrequentItemSet:
        subsets = powerset(itemSet)
        itemSetSup = FrequentItemSet[itemSet]
        for s in subsets:
            remain = itemSet.difference(s)
            confidence = itemSetSup / FrequentItemSet[frozenset(s)]
            lift = confidence *total / FrequentItemSet[frozenset(remain)]
            if(confidence >= minConf): AssociationRule.append([set(s), set(remain),itemSetSup/total, confidence,lift])
    return AssociationRule

def preprocessData(input_data, a):
    itemSetList, frequency = changeToList(input_data)
    minSup = len(itemSetList) * a.min_sup
    total = len(itemSetList)
    fpTree, headerTable = CreateTree(itemSetList, frequency, minSup)
    freqItems = {}
    FindFrequentItemSet(headerTable, minSup, set(), freqItems)
    return FindAssociationRule(freqItems, a.min_conf,total)