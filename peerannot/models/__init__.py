from .MV import MV
from .Soft import Soft
from .DS import Dawid_Skene
from .DS_clust import Dawid_Skene_clust
from .GLAD import GLAD
from .WAUM import WAUM
from .WAUM_redundant import WAUM_redundant

agg_strategies = {
    "MV": MV,
    "NaiveSoft": Soft,
    "DS": Dawid_Skene,
    "DSWC": Dawid_Skene_clust,
    "GLAD": GLAD,
}
