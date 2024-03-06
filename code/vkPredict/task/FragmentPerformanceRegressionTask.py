import model.ModelBuilder as ModelBuilder

class FragmentPerformanceRegressionTask:
    def __init__(self, model):
        self.model = model

    def criterion(self):
        pass

    def setup_model(self, *args, **kwargs):
        self.model = ModelBuilder.build_model(*args, **kwargs)
    