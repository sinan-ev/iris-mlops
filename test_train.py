from train import train

import joblib

def test_train():
    model=joblib.load("model/model.pkl")
    assert model is not None
    print('tested')

if __name__ == "__main__":
    test_train()

