from detectors.ghostbuster.model import GhostbusterDetector

if __name__ == '__main__':
    model = GhostbusterDetector('../ghostbuster/bin')
    print(model.predict_proba('Hello biden!'))
