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
    
   import torchsort

class MultipleNegativesRankingLossWithSpearman(nn.Module):

    def __init__(self, model: SentenceTransformer, scale: float = 20.0, similarity_fct = util.cos_sim, spearmans = False, spearmans_boost = 1, regularization_strength = 1):
        """
        :param model: SentenceTransformer model
        :param scale: Output of similarity function is multiplied by scale value
        :param similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot product (and then set scale to 1)
        """
        super(MultipleNegativesRankingLossWithSpearman, self).__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.BCELoss()
        self.softmax = nn.Softmax(dim=1)
        if spearmans:
          ce_name = 'msmarco-electra-cross-encoder-v26/validation'
          with open("models/train_set_result_distilation/" + ce_name +"/sorted.scores.groups.of.32.10.CE.hard.negatives.as.triples.batch.size.4.shuffled.pkl", "rb") as h:
            self.ce_scores = pickle.load(h)
            self.spearmans = True
            self.spearmans_boost = spearmans_boost
            self.regularization_strength = regularization_strength
          
    #SR is between -1 and 1. As loss, between 0 and 2.
    def _spearmanr_loss(self, pred, target, **kw):
      pred = torchsort.soft_rank(pred, **kw)
      target = torchsort.soft_rank(target, **kw)
      pred = pred - pred.mean()
      pred = pred / pred.norm()
      target = target - target.mean()
      target = target / target.norm()

      spearman_r = (pred * target).sum()
      as_loss = (-spearman_r + 1) / 2
      return as_loss


    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        embeddings_a = reps[0]
        embeddings_b = torch.cat(reps[1:])
        num_passages_per_query = len(embeddings_a) * 2

        # keep scores as a matrix, will use it for the spearmans
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        
        #scores_T = torch.transpose(scores, 0, 1)
        
        # softmax each row of the matrix and flatten it to a vector
        scores_softmaxed = self.softmax(scores)    
        for_binary = torch.flatten(scores_softmaxed)


        final_preds = for_binary


        #create the labels
        labels_for_binary = []
        for i in range(len(scores_softmaxed)):
          temp = [0] * num_passages_per_query
          temp[i] = 1
          labels_for_binary += temp


        labels_ground_truth_for_binary = torch.tensor(labels_for_binary, dtype=torch.float, device=scores.device)  # 0 or 1 for irrelevant/relevant passages.
        
        ce = self.cross_entropy_loss(final_preds, labels_ground_truth_for_binary)

        if self.spearmans:
          batch_ce_scores = self.ce_scores[labels[0]]

          targets = []

          margin = 99999 #add this margin to the relevat pasage to correct the cross-encoder
          relevant_pos = 0

          # create a matrix of target values from cross_encoder scores, same shape as scores matrix.
          for i in range(0,int(len(batch_ce_scores)/8)):
            sub_list = batch_ce_scores[i*8:(i+1)*8] 

            sub_list = [i*1000 for i in sub_list]
            sub_list[relevant_pos] += margin

            relevant_pos += 1

            targets.append(sub_list)
          targets = torch.tensor(targets).cuda()


          s_loss = self._spearmanr_loss(scores, targets, regularization_strength=self.regularization_strength)
        
        loss = ce + s_loss 

        return loss
