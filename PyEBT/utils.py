import numpy as np

def lin_impute(t, y, mindex):
    """
    Perform linear interpolation to fill missing values in the data.
    """
    length = len(y)
    index = np.arange(length)
    mindex = np.sort(mindex)
    oindex = np.setdiff1d(index, mindex)
    yin = np.full(length, np.nan)
    yin[oindex] = y[oindex]
    
    if mindex[0] == 0:
        index_imputed = np.arange(0, oindex[0])
        for j in index_imputed:
            yin[j] = lextra(p1=(t[oindex[0]], y[oindex[0]]), 
                            p2=(t[oindex[1]], y[oindex[1]]), 
                            x=t[j])

    for i in range(len(oindex) - 1):
        if oindex[i + 1] - oindex[i] > 1:
            index_imputed1 = np.arange(oindex[i], oindex[i + 1] + 1)
            index_imputed2 = index_imputed1[1:-1]
            for j in index_imputed2:
                yin[j] = linter(p1=(t[index_imputed1[0]], y[index_imputed1[0]]), 
                                p2=(t[index_imputed1[-1]], y[index_imputed1[-1]]), 
                                x=t[j])
    
    if (length - oindex[-1] - 1) > 0:
        index_imputed = np.arange(oindex[-1] + 1, length)
        for j in index_imputed:
            yin[j] = lextra(p1=(t[oindex[-2]], y[oindex[-2]]), 
                            p2=(t[oindex[-1]], y[oindex[-1]]), 
                            x=t[j])
    return yin


def linter(p1, p2, x):
    """
    Linear interpolation between two points p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    m = abs(x1 - x)
    n = abs(x2 - x)
    y = (m * y2 + n * y1) / (m + n)
    return y


def lextra(p1, p2, x):
    """
    Linear extrapolation between two points p1 and p2.
    """
    x1, y1 = p1
    x2, y2 = p2
    m = abs(x1 - x)
    n = abs(x2 - x)
    y = (m * y2 - n * y1) / (m - n)
    return y


def left_const(y, mindex):
    """
    Left constant interpolation to fill missing values in the data.
    """
    length = len(y)
    index = np.arange(length)
    mindex = np.sort(mindex)
    oindex = np.setdiff1d(index, mindex)
    yin = np.full(length, np.nan)
    yin[oindex] = y[oindex]
    
    if mindex[0] == 0:
        index_imputed = np.arange(0, oindex[0])
        yin[index_imputed] = y[oindex[0]]
    
    for i in range(len(oindex) - 1):
        if oindex[i + 1] - oindex[i] > 1:
            index_imputed = np.arange(oindex[i], oindex[i + 1] + 1)[1:-1]
            yin[index_imputed] = y[oindex[i]]
    
    if (length - oindex[-1] - 1) > 0:
        index_imputed = np.arange(oindex[-1] + 1, length)
        yin[index_imputed] = y[oindex[-1]]
    
    return yin


def right_const(y, mindex):
    """
    Right constant interpolation to fill missing values in the data.
    """
    length = len(y)
    index = np.arange(length)
    mindex = np.sort(mindex)
    oindex = np.setdiff1d(index, mindex)
    yin = np.full(length, np.nan)
    yin[oindex] = y[oindex]
    
    if mindex[0] == 0:
        index_imputed = np.arange(0, oindex[0])
        yin[index_imputed] = y[oindex[0]]
    
    for i in range(len(oindex) - 1):
        if oindex[i + 1] - oindex[i] > 1:
            index_imputed = np.arange(oindex[i], oindex[i + 1] + 1)[1:-1]
            yin[index_imputed] = y[oindex[i + 1]]
    
    if (length - oindex[-1] - 1) > 0:
        index_imputed = np.arange(oindex[-1] + 1, length)
        yin[index_imputed] = y[oindex[-1]]
    
    return yin