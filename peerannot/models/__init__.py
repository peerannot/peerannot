from .MV import MV
from .Soft import Soft
from .DS import Dawid_Skene
from .DS_clust import Dawid_Skene_clust
from .GLAD import GLAD
from .WAUM import WAUM
from .WAUM_stacked import WAUM_stacked
from .CoNAL import CoNAL

agg_strategies = {
    "MV": MV,
    "NaiveSoft": Soft,
    "DS": Dawid_Skene,
    "DSWC": Dawid_Skene_clust,
    "GLAD": GLAD,
    "CoNAL": CoNAL,
}
