from peerannot.helpers.converters import Converter


class CrowdModel:
    def __init__(self, answers):
        self.converter = Converter(answers)
        self.answers = self.converter.transform()
