import paddle

def calc_mean_std(feat, eps=1e-5):
    size = feat.shape
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.reshape([N, C, -1])
    feat_var = paddle.var(feat_var, axis=2) + eps
    feat_std = paddle.sqrt(feat_var)
    feat_std = feat_std.reshape([N, C, 1, 1])
    feat_mean = feat.reshape([N, C, -1])
    feat_mean = paddle.mean(feat_mean, axis=2)
    feat_mean = feat_mean.reshape([N, C, 1, 1])
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.shape
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.shape[:2] == style_feat.shape[:2])
    size = content_feat.shape
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat -
                       content_mean.expand(size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def save_params(net, net_name, save_dir, idx):
    state_dict = net.state_dict()
    paddle.save(state_dict, '%s/%s_epoch_%s.pdparams' % (save_dir, net_name, idx))



