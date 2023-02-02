from .MV import MV
from .NaiveSoft import NaiveSoft
from .DS import Dawid_Skene
from .DS_clust import Dawid_Skene_clust
from .GLAD import GLAD
from .WAUM_perworker import WAUM_perworker
from .WAUM import WAUM
from .CoNAL import CoNAL
from .Crowdlayer import Crowdlayer
from .AUM import AUM
from .WDS import WDS

agg_strategies = {
    "MV": MV,
    "NaiveSoft": NaiveSoft,
    "DS": Dawid_Skene,
    "DSWC": Dawid_Skene_clust,
    "GLAD": GLAD,
    "WDS": WDS,
    "CoNAL": CoNAL,
    "CrowdLayer": Crowdlayer,
    "AUM": AUM,
}
