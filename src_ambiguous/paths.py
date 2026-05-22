from enum import Enum

class DATASETS(Enum):
    BIRDdev = "../datasets/BIRDdev/"
    BIRDdev_ambiguos = "../datasets/BIRDdev-ambiguos/"
    SPIDERdev1 = "../datasets/SPIDERdev1.0/"
    SPIDERdev1_ambiguos = "../datasets/SPIDERdev1.0-ambiguos/"

class RESULTS(Enum):
    DB_RETRIEVAL = '../results/DB_retrieval/'
    TAB_RETRIEVAL = '../results/TAB_retrieval/'
    ATT_RETRIEVAL = '../results/ATT_retrieval/'