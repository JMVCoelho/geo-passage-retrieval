import warnings

from torch.nn import PairwiseDistance
from torch.nn import Module
import torch.nn.functional as F
import torch.nn._reduction as _Reduction

from torch import Tensor
from typing import Callable, Optional

import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction


class _WeightedLoss(_Loss):
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.weight: Optional[Tensor]

class CustomWeightBCELoss(_WeightedLoss):

    __constants__ = ['reduction']

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(CustomWeightBCELoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target, weight=weight, reduction=self.reduction)


class CustomWeightBCELossWithLogits(_WeightedLoss):

    __constants__ = ['reduction']

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(CustomWeightBCELossWithLogits, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor, weight: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target, weight=weight, reduction=self.reduction)

class MultipleNegativesRankingLossBinary(nn.Module):

    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLossBinary, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.BCELoss()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])


        num_passages_per_query = len(embeddings_a) * 2

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        #scores_T = torch.transpose(scores, 0, 1)


        scores = self.softmax(scores)
        #scores_T = self.softmax(scores_T)
      
        #scores_T_T = torch.transpose(scores_T, 0, 1)
    
        for_binary = torch.flatten(scores)
        #for_binary_2 = torch.flatten(scores_T_T)

        #final_preds = (for_binary + for_binary_2) / 2
        final_preds = for_binary

        labels = []
        for i in range(len(scores)):
          temp = [0] * num_passages_per_query
          temp[i] = 1
          labels += temp




        labels = torch.tensor(labels, dtype=torch.float, device=scores.device)  # 0 or 1 for irrelevant/relevant passages.

        
        return self.cross_entropy_loss(final_preds, labels)

import torch
from torch import nn, Tensor
from typing import Iterable, Dict
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

class MultipleNegativesRankingLossBinaryCustomWeight(nn.Module):

    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLossBinaryCustomWeight, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        #self.cross_entropy_loss = nn.BCELoss() 
        self.cross_entropy_loss = CustomWeightBCELoss()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])

        weights = labels

        num_passages_per_query = len(embeddings_a) * 2

        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        #scores_T = torch.transpose(scores, 0, 1)


        scores = self.softmax(scores)
        #scores_T = self.softmax(scores_T)
      
        #scores_T_T = torch.transpose(scores_T, 0, 1)
    
        for_binary = torch.flatten(scores)
        #for_binary_2 = torch.flatten(scores_T_T)

        #final_preds = (for_binary + for_binary_2) / 2
        final_preds = for_binary

        labels = []
        for i in range(len(scores)):
          temp = [0] * num_passages_per_query
          temp[i] = 1
          labels += temp

        w = []
        for j in range(len(scores)):
          temp = [weights[j].item()] * num_passages_per_query
          w += temp

        labels = torch.tensor(labels, dtype=torch.float, device=scores.device)  # 0 or 1 for irrelevant/relevant passages.

        w = torch.tensor(w, dtype=torch.float, device=scores.device)
        
        return self.cross_entropy_loss(final_preds, labels, weight=w)