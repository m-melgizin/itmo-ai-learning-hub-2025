from models import *
import pandas as pd

if __name__ == '__main__':
    train_data = pd.read_csv('data/train.csv')
    val_data = pd.read_csv('data/val.csv')
    model = VectorizerWithClassifier(TfidfVectorizer(), LogisticRegression())
    metrics = model.train(train_data, val_data)

    print(model)
    print(metrics)
