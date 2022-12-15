"""
=================================
Parent template to all strategies
=================================
"""

from peerannot.helpers.converters import Converter


class CrowdModel:
    def __init__(self, answers):
        self.converter = Converter(answers)
        self.answers = self.converter.transform()
        self.answers = dict(
            sorted({int(k): v for k, v in self.answers.items()}.items())
        )
