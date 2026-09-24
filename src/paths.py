from enum import Enum

class DATASETS(Enum):
    BIRDdev = "../datasets/BIRDdev/"
    BIRDdev_ambiguous = "../datasets/BIRDdev-ambiguous/"
    SPIDERdev1 = "../datasets/SPIDERdev1.0/"
    SPIDERdev1_ambiguous = "../datasets/SPIDERdev1.0-ambiguous/"
    BIRDtrain = "../datasets/BIRDtrain/"
    BIRDtrain_ambiguous = "../datasets/BIRDtrain-ambiguous/"
    BEAVER = "../datasets/BEAVER/"
    BEAVER_ambiguous = "../datasets/BEAVER-ambiguous/"
    ARCHER = "../datasets/ARCHER/"
    ARCHER_ambiguous = "../datasets/ARCHER-ambiguous/"
    SPIDERtrain = "../datasets/SPIDERtrain/"
    SPIDERtrain_ambiguous = "../datasets/SPIDERtrain-ambiguous/"
    SQALE3 = "../datasets/SQALE3/"
    SQALE3_ambiguous = "../datasets/SQALE3-ambiguous/"

class RESULTS(Enum):
    DB_RETRIEVAL = '../results/DB_retrieval/'
    TAB_RETRIEVAL = '../results/TAB_retrieval/'
    ATT_RETRIEVAL = '../results/ATT_retrieval/'