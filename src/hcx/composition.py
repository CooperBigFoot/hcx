from torch import nn

from hcx.batch import Batch
from hcx.output import Forecast
from hcx.protocol import FeatureExtractor, ForecastModel


class ComposedModel(nn.Module):
    def __init__(self, extractor: FeatureExtractor, forecaster: ForecastModel) -> None:
        super().__init__()
        self.extractor = extractor
        self.forecaster = forecaster

    @property
    def consumed_quadrants(self) -> object:
        return getattr(self.extractor, "consumed_quadrants", None)

    def forward(self, batch: Batch) -> Forecast:
        return self.forecaster.forward(self.extractor.forward(batch))
