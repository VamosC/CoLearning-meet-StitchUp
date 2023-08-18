import torch


def union_label(lb_a, lb_b):

    return torch.where((lb_a > 0)|(lb_b > 0),
                       torch.ones_like(lb_a),
                       torch.zeros_like(lb_a))


def find_common_label(lb_a, lb_b):

    return torch.where((lb_a > 0)&(lb_b > 0),
                       torch.ones_like(lb_a),
                       torch.zeros_like(lb_a))
