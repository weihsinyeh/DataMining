from typing import List
from itertools import chain, combinations
from collections import defaultdict
global min_sup,min_conf,total
import math
import gc
def subsets(arr):
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

def ItemsWithMinSupport(itemSet, transactionList, FreqentSet):
    _itemSet = set()
    localSet = defaultdict(int)
    for item in itemSet:
        for transaction in transactionList:
            if item.issubset(transaction):
                FreqentSet[item] += 1
                localSet[item] += 1
    for item, support in localSet.items():

        if support >= min_sup:  
            _itemSet.add(item)
    return _itemSet

def joinSet(itemSet, length):
    joinSet = set()
    for i in itemSet:
        for j in itemSet:
            if len(i.union(j)) == length: joinSet.add(i.union(j))
    return joinSet

def Generate_1_ItemSet(input_data):
    transactionList = list()
    itemSet = set()
    for record in input_data:
        transaction = set(record)
        transactionList.append(transaction)
        for item in transaction:  itemSet.add(frozenset([item]))  # Generate 1-itemSets
    return itemSet, transactionList
    
def Apriori(input_data):
    itemSet, transactionList = Generate_1_ItemSet(input_data)
    FreqentSet = defaultdict(int)
    largeSet = dict()
        
    oneCandiateSet = ItemsWithMinSupport(itemSet, transactionList, FreqentSet)
    currentLargeSet = oneCandiateSet
    k = 2
    while currentLargeSet != set([]):
        largeSet[k - 1] = currentLargeSet
        currentLargeSet = joinSet(currentLargeSet, k)
        currentCSet = ItemsWithMinSupport(currentLargeSet, transactionList, FreqentSet)
        currentLargeSet = currentCSet
        k += 1
    toRetItems = []
    for key, value in largeSet.items():
        toRetItems.extend([(tuple(item), FreqentSet[item] ) for item in value])
    '''
    for item in freqSet:
        print(item,':',freqSet[item])
        if freqSet[item] >=  min_sup: 
            print(item,':',freqSet[item])
    '''
    AssociationRules = []
    for key, value in list(largeSet.items())[1:]:
        for item in value:
            _subsets = frozenset([x for x in subsets(item)])
            for element in _subsets:
                remain = item.difference(element)    
                if len(remain) > 0:
                    confidence = FreqentSet[frozenset(item)]/ FreqentSet[frozenset(element)]
                    if confidence >= min_conf:
                        support = FreqentSet[frozenset(item)] /total
                        lit     = float(confidence)/ (FreqentSet[frozenset(remain)]/total)
                        AssociationRules.append((frozenset(element), frozenset(remain),support,confidence,lit))
    return AssociationRules 
def preprocessData(input_data: List[List[str]], a) -> List[List[str]]:
    global min_sup , min_conf,total
    min_conf = a.min_conf
    map ,countitem = {},{}
    for data in input_data:
        if data[0] not in map       : map[data[0]] = []
        if data[2] not in countitem : countitem[data[2]] = 0
        countitem[data[2]]  += 1  
        map[data[0]].append(data[2])
    total = map.__len__()  
    min_sup = math.ceil(a.min_sup* total)
    map = {}
    for data in input_data:
        if countitem[data[2]] >= min_sup : 
            if data[0] not in map : map[data[0]] = []
            map[data[0]].append(data[2])
    input_data = []
    for key, value in map.items(): input_data.append(value)
    AssociationRules = Apriori(input_data)
    return AssociationRules