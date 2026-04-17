from .agg_deep.CoNAL import CoNAL
from .agg_deep.Crowdlayer import Crowdlayer
from .aggregation.dawid_skene import DawidSkene
from .aggregation.DS_clust import DawidSkeneClust
from .aggregation.GLAD import GLAD
from .aggregation.IWMV import IWMV
from .aggregation.majority_voting import MajorityVoting
from .aggregation.NaiveSoft import NaiveSoft
from .aggregation.plantnet import PlantNet
from .aggregation.twothird import TwoThird
from .aggregation.Wawa import Wawa
from .aggregation.WDS import WDS
from .identification.AUM import AUM
from .identification.entropy import Entropy
from .identification.krippendorff_alpha import Krippendorff_Alpha
from .identification.Spam_score import Spam_Score
from .identification.trace_confusion import Trace_confusion
from .identification.WAUM import WAUM
from .identification.WAUM_perworker import WAUM_perworker

agg_strategies = {
    "majority_voting": MajorityVoting,
    "NaiveSoft": NaiveSoft,
    "dawid_skene": DawidSkene,
    "DSWC": DawidSkeneClust,
    "GLAD": GLAD,
    "WDS": WDS,
    "PlantNet": PlantNet,
    "TwoThird": TwoThird,
    "IWMV": IWMV,
    "Wawa": Wawa,
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
    "trace_confusion": Trace_confusion,
    "spam_score": Spam_Score,
    "krippendorffAlpha": Krippendorff_Alpha,
}
