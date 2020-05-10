import torch
import torch.nn as nn


class DenseNet121Extractor(nn.Module):
    def __init__(self, submodule, extracted_layer):
        super(DenseNet121Extractor, self).__init__()
        self.submodule = submodule
        self.extracted_layer = extracted_layer
        self.features = self._get_features()

    def forward(self, x):
        x = self.features(x)
        return x

    def _get_features(self):
        if self.extracted_layer == 1:               # end of denseblock1
            modules = list(self.submodule.features.children())[:5]
        elif self.extracted_layer == 2:             # end of transition1
            modules = list(self.submodule.features.children())[:6]
        elif self.extracted_layer == 3:             # end of denseblock2
            modules = list(self.submodule.features.children())[:7]
        elif self.extracted_layer == 4:             # end of transition2
            modules = list(self.submodule.features.children())[:8]
        elif self.extracted_layer == 5:             # end of denseblock3
            modules = list(self.submodule.features.children())[:9]
        elif self.extracted_layer == 6:             # end of transition3
            modules = list(self.submodule.features.children())[:10]
        else:                                       # end of norm5 following denseblock4
            modules = list(self.submodule.features.children())[:12]

        features = nn.Sequential(*modules)

        return features


class ResNetExtractor(nn.Module):
    def __init__(self, submodule, extracted_layer, net_model):
        super(ResNetExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layer = extracted_layer
        self.net_model = net_model

    def forward(self, x):
        if self.extracted_layer == 1:                       # end of the first plain max-pool
            modules = list(self.submodule.children())[:4]
        elif self.extracted_layer == 2:                     # end of layer1
            modules = list(self.submodule.children())[:5]
        elif self.extracted_layer == 3:                     # end of layer2
            modules = list(self.submodule.children())[:6]
        elif self.extracted_layer == 4:                     # the inner third module of layer3
            modules = list(self.submodule.children())[:6]
            third_module = list(self.submodule.children())[6]
            if self.net_model == 'resnet50':
                third_module_modules = list(third_module.children())[:3]  # take the first three inner modules
            else:   # net_model is 'resnet101'
                third_module_modules = list(third_module.children())[:12]  # take the first three inner modules

            third_module = nn.Sequential(*third_module_modules)
            modules.append(third_module)
        elif self.extracted_layer == 5:                     # end of layer3
            modules = list(self.submodule.children())[:7]
        elif self.extracted_layer == 6:                     # end of layer4
            modules = list(self.submodule.children())[:8]
        else:                                               # end of the last avg-pool
            modules = list(self.submodule.children())[:9]

        self.submodule = nn.Sequential(*modules)
        x = self.submodule(x)
        return x


class AlexNetVGG16Extractor(nn.Module):
    def __init__(self, original_model, extracted_layer, net_model):
        super(AlexNetVGG16Extractor, self).__init__()
        self.extracted_layer = extracted_layer
        self.net_model = net_model
        self.features = original_model.features
        self.avgpool = original_model.avgpool
        self.classifier = original_model.classifier

    def forward(self, x):
        if self.extracted_layer < 6:
            self.features = self._get_features()
            x = self.features(x)
        else:
            self.classifier = self._get_classifier()
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x

    def _get_features(self):
        index = self._find_index()
        features = nn.Sequential(
            # stop at the layer
            *list(self.features.children())[:index]
        )
        return features

    def _get_classifier(self):
        index = self._find_index()
        classifier = nn.Sequential(
            # stop at the layer
            *list(self.classifier.children())[:index]
        )
        return classifier

    def _find_index(self):
        if self.net_model == 'alexnet':
            switcher = {
                1: 3,  # from features
                2: 6,
                3: 8,
                4: 10,
                5: 13,
                6: 3,  # from classifier
                7: 6
            }
        else:   # net_model is 'vgg16_bn'
            switcher = {
                1: 7,  # from features
                2: 14,
                3: 24,
                4: 34,
                5: 44,
                6: 2,  # from classifier
                7: 5
            }

        return switcher.get(self.extracted_layer)
