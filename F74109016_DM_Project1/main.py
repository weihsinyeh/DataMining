"""
The entry point of the program

First of all, you don't have to follow the structure of this file,
but please make sure that: (1)we can run your code by running this file,
(2)it can generate the output files, and (3)it can accept the command line arguments.

Please implement the `apriori` and `fp_growth` functions in
the `my_cool_algorithms.py` file (or any module name you prefer).

The `input_data` is a list of lists of integers. Each inner list
is in the form of [transaction_id, transaction_id, item_id].
For example, the following input data contains 2 transactions,
transaction 1 contains 2 items 9192, 31651;
transaction 2 contains 2 items 26134, 57515.

[
    [1, 1, 9192],
    [1, 1, 31651],
    [2, 2, 26134],
    [2, 2, 57515],
]


The `a` is a `Namespace` object that contains the following attributes:
    - dataset: the name of the dataset
    - min_sup: the minimum support
    - min_conf: the minimum confidence
you can access them by `a.dataset`, `a.min_sup`, `a.min_conf`.
"""
from pathlib import Path
from typing import List
import args
import config
import utils
from utils import l
import matplotlib.pyplot as plt
import time
import FP_growth
import Apriori
def main():
    # Parse command line arguments
    a = args.parse_args()
    l.info(f"Arguments: {a}")
    #min_sup_list =  [0.05,0.075,0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275, 0.3] #[0.2 ,  0.2,0.2,  0.2, 0.2,  0.2,0.2,  0.2, 0.2,  0.2, 0.2]
    #min_conf_list =  [0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05, 0.05,0.05, 0.05]
    #Apriori_elapsed_time = []
    #FP_growth_elapsed_time = []
    #for i in range(0,len(min_sup_list)):
    #    a.min_sup = min_sup_list[i]
    #    a.min_conf = min_conf_list[i]
    # Load dataset, the below io handles ibm dataset
    input_data: List[List[str]] = utils.read_file(config.IN_DIR / a.dataset)
    filename = Path(a.dataset).stem
    print("min-sup : ",a.min_sup," min-conf : ",a.min_conf)
    ################### Apriori ###################
    start_time = time.time()
    apriori_out = Apriori.preprocessData(input_data, a)
    end_time = time.time()
    #Apriori_elapsed_time.append(end_time - start_time)
    print("Apriori - elapsed_time : ",end_time - start_time ,"sec")
    print("Apriori - Association-Rules:",apriori_out.__len__())
    apriori_out = sorted(apriori_out, key=lambda x: x[4])
    utils.write_file(
        data=apriori_out,
        filename=config.OUT_DIR / f"{filename}-apriori-{a.min_sup}-{a.min_conf}.csv"
    )
    ################### FP_growth ###################
    start_time = time.time()
    fp_growth_out = FP_growth.preprocessData(input_data,a)
    end_time = time.time()
    #FP_growth_elapsed_time.append(end_time - start_time)
    print("FP-growth - elapsed_time : ",end_time - start_time,"sec") 
    print("FP-growth - Association-Rules:",fp_growth_out.__len__())
    utils.write_file(
        data=fp_growth_out,
        filename=config.OUT_DIR / f"{filename}-fp_growth-{a.min_sup}-{a.min_conf}.csv"
    )
    #print(len(Apriori_elapsed_time))
    #print(len(FP_growth_elapsed_time))
    ##### draw graph Fixed Confidence = 0.2 #####
    '''
    plt.figure(figsize=(10, 6))
    plt.plot(min_sup_list, Apriori_elapsed_time, label='Apriori',linewidth=10)
    plt.plot(min_sup_list, FP_growth_elapsed_time, label='FP-growth',linewidth=10)
    plt.xlabel('Min -  Support')
    plt.ylabel('Elapsed Time (seconds)')
    plt.suptitle('Apriori vs. FP-growth Elapsed Time')
    plt.title('Min - Confidence = 0.05')
    plt.legend()
    plt.grid(True)
    plt.savefig('Apriori vs. FP-growth Elapsed Time(fixed cofidence).png')
    plt.show()
    
     ##### draw graph Fixed support = 0.2 #####
    plt.figure(figsize=(10, 6))
    plt.plot(min_conf_list, Apriori_elapsed_time, label='Apriori',linewidth=10)
    plt.plot(min_conf_list, FP_growth_elapsed_time, label='FP-growth',linewidth=10)
    plt.xlabel('Min -  Confidence')
    plt.ylabel('Elapsed Time (seconds)')
    plt.suptitle('Apriori vs. FP-growth Elapsed Time')
    plt.title('Min - Support = 0.05')
    plt.legend()
    plt.grid(True)
    plt.savefig('Apriori vs. FP-growth Elapsed Time(fixed support).png')
    plt.show()
    '''
if __name__ == "__main__":
    main()

