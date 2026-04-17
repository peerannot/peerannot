from .agg_deep.CoNAL import CoNAL
from .agg_deep.Crowdlayer import Crowdlayer
from .aggregation.dawid_skene import DawidSkene
from .aggregation.dawid_skene_clust import DawidSkeneClust
from .aggregation.glad import GLAD
from .aggregation.iwmv import IWMV
from .aggregation.majority_voting import MajorityVoting
from .aggregation.naive_soft import NaiveSoft
from .aggregation.plantnet import PlantNet
from .aggregation.twothird import TwoThird
from .aggregation.wawa import Wawa
from .aggregation.wds import WDS
from .identification.AUM import AUM
from .identification.entropy import Entropy
from .identification.krippendorff_alpha import Krippendorff_Alpha
from .identification.Spam_score import Spam_Score
from .identification.trace_confusion import Trace_confusion
from .identification.WAUM import WAUM
from .identification.WAUM_perworker import WAUM_perworker

agg_strategies = {
    "majority_voting": MajorityVoting,
    "naive_soft": NaiveSoft,
    "dawid_skene": DawidSkene,
    "dawid_skene_clust": DawidSkeneClust,
    "glad": GLAD,
    "wds": WDS,
    "plantnet": PlantNet,
    "twothird": TwoThird,
    "iwmv": IWMV,
    "wawa": Wawa,
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
