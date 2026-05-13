import torch
import torch.nn.functional as F


def calculate_SPO_loss(train_prefered_logprob, train_disprefered_logprob,
                       ref_prefered_logprob, ref_disprefered_logprob, beta=0.5):
    prefered_relative_logprob = train_prefered_logprob - ref_prefered_logprob
    disprefered_relative_logprob = train_disprefered_logprob - ref_disprefered_logprob

    logits = beta * (prefered_relative_logprob - disprefered_relative_logprob)
    loss = -F.logsigmoid(logits).mean()

    reward_accuracies = (prefered_relative_logprob > disprefered_relative_logprob).float()
    reward_margins = prefered_relative_logprob - disprefered_relative_logprob

    return loss, prefered_relative_logprob, disprefered_relative_logprob, reward_accuracies, reward_margins
