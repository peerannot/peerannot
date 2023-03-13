from .aggregation.MV import MV
from .aggregation.NaiveSoft import NaiveSoft
from .aggregation.DS import Dawid_Skene
from .aggregation.DS_clust import Dawid_Skene_clust
from .aggregation.GLAD import GLAD
from .aggregation.WDS import WDS
from .identification.WAUM_perworker import WAUM_perworker
from .identification.WAUM import WAUM
from .identification.AUM import AUM
from .identification.entropy import Entropy
from .agg_deep.CoNAL import CoNAL
from .agg_deep.Crowdlayer import Crowdlayer

agg_strategies = {
    "MV": MV,
    "NaiveSoft": NaiveSoft,
    "DS": Dawid_Skene,
    "DSWC": Dawid_Skene_clust,
    "GLAD": GLAD,
    "WDS": WDS,
}

agg_deep_strategies = {
    "CoNAL": CoNAL,
    "CrowdLayer": Crowdlayer,
}

identification_strategies = {
    "AUM": AUM,
    "WAUM_perworker": WAUM_perworker,
    "WAUM": WAUM,
    "entropy": Entropy,
}
