def compute_errors(x, reconstruction):
    """逐样本、逐原始特征的平方误差。"""
    return (x - reconstruction[:, :x.shape[1]]).square()


def compute_scores(x, reconstruction):
    """只按原始数据特征的平均平方误差打分，分数越高越异常。"""
    return compute_errors(x, reconstruction).mean(dim=1)
