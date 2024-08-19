# 
import torch
import torch.nn.functional as F

def avg_max_reduce_hw_helper(x, use_concat=True):
    assert not isinstance(x, (list, tuple))
    avg_pool = F.adaptive_avg_pool2d(x, 1)
    max_pool = F.adaptive_max_pool2d(x, 1)

    if use_concat:
        res = torch.cat([avg_pool, max_pool], dim=1)
    else:
        res = [avg_pool, max_pool]
    return res

def avg_max_reduce_hw(x):
    # Reduce hw by avg and max
    # Return cat([avg_pool_0, avg_pool_1, ..., max_pool_0, max_pool_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_hw_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_hw_helper(x[0])
    else:
        res_avg = []
        res_max = []
        for xi in x:
            avg, max = avg_max_reduce_hw_helper(xi, False)
            res_avg.append(avg)
            res_max.append(max)
        res = res_avg + res_max
        return torch.cat(res, dim=1)


def mean_max_reduce_channel_helper(x, use_concat=True):
    # Reduce channel by mean and max, only support single input
    assert not isinstance(x, (list, tuple))
    mean_value = torch.mean(x, dim=1, keepdim=True)
    max_value = torch.max(x, dim=1, keepdim=True).values

    if use_concat:
        res = torch.cat([mean_value, max_value], dim=1)
    else:
        res = [mean_value, max_value]
    return res

def mean_max_reduce_channel(x):
    # Reduce channel by mean and max
    # Return cat([mean_ch_0, max_ch_0, mean_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return mean_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return mean_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(mean_max_reduce_channel_helper(xi, False))
        return torch.cat(res, dim=1)