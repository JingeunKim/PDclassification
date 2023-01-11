from sklearn.model_selection import train_test_split


def preprocess_inputscv_FS(df, i):
    df = df.copy()
    y = df['class']
    X = df.drop(['class'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=i)

    return X_train, X_test, y_train, y_test


def preprocess_inputscv(df, i):
    df = df.copy()

    y = df.loc[17580]
    x = df.iloc[:-1]
    X = x.transpose()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=i)

    return X_train, X_test, y_train, y_test

def preprocess_inputscv_IG(df, i):
    df = df.copy()
    df = df.transpose()

    y = df.loc[75]
    x = df.iloc[:-1]
    X = x.transpose()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=i)

    return X_train, X_test, y_train, y_test

def preprocess_inputscv_wolf(df, i):
    df = df.copy()
    df = df.transpose()

    y = df.loc[8359]
    x = df.iloc[:-1]
    X = x.transpose()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=i)

    return X_train, X_test, y_train, y_test