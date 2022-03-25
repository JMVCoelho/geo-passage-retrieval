from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
import logging
import os
from typing import Dict, Type, Callable, List
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import SentenceEvaluator
import torch.nn.functional as F
import pandas as pd

logger = logging.getLogger(__name__)

def generate_eval_samples_dict():
  dev_samples = dict()

  path_qrels = 'runs/qrels/train_and_validation/validation/qrels.st.validation.tsv'
  relevants = pd.read_csv(path_qrels, sep='\t', names=['qId', 'ign1', 'pId', 'ign2'])

  def isRelevant(pId, qId):
    rels = relevants.query('qId == @qId')['pId'].values
    if pId in rels:
      return True
    else:
      return False

  df = pd.read_csv("geographic_reranking/train_and_validation/validation/runs/bm25-top25-with-geo-distance.tsv", sep='\t', names=['qId', 'pId', 'pos', 'distance'])
  df = df[['qId', 'pId', 'pos']]

  all_queries = pd.read_csv('queries.train.tsv', sep='\t', names=['qId', 'text'])
  all_queries = dict(zip(all_queries.qId, all_queries.text))

  corpus = dict(zip(collection.pId, collection.text))

  val_queries =  list(set(df['qId'].to_list()))

  print("Queries: ", len(val_queries))

  for qId in val_queries:
    dev_samples[qId] = dict()
    dev_samples[qId]['query'] = all_queries[qId]
    dev_samples[qId]['positive'] = []
    dev_samples[qId]['negative'] = []

    top25 = df.query('qId == @qId')['pId'].values 
    for pId in top25:
      if isRelevant(pId, qId):
        dev_samples[qId]['positive'].append(corpus[pId])
      else:
        dev_samples[qId]['negative'].append(corpus[pId])
    
    dev_samples[qId]['negative'] = set(dev_samples[qId]['negative'])
    dev_samples[qId]['positive'] = set(dev_samples[qId]['positive'])

  return dev_samples

class ModifiedCrossEncoder():
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None, device:str = None, tokenizer_args:Dict = {},
                 default_activation_function = None):
        """
        A CrossEncoder takes exactly two sentences / texts as input and either predicts
        a score or label for this sentence pair. It can for example predict the similarity of the sentence pair
        on a scale of 0 ... 1.
        It does not yield a sentence embedding and does not work for individually sentences.
        :param model_name: Any model name from Huggingface Models Repository that can be loaded with AutoModel. We provide several pre-trained CrossEncoder models that can be used for common tasks
        :param num_labels: Number of labels of the classifier. If 1, the CrossEncoder is a regression model that outputs a continous score 0...1. If > 1, it output several scores that can be soft-maxed to get probability scores for the different classes.
        :param max_length: Max length for input sequences. Longer sequences will be truncated. If None, max length of the model will be used
        :param device: Device that should be used for the model. If None, it will use CUDA if available.
        :param tokenizer_args: Arguments passed to AutoTokenizer
        :param default_activation_function: Callable (like nn.Sigmoid) about the default activation function that should be used on-top of model.predict(). If None. nn.Sigmoid() will be used if num_labels=1, else nn.Identity()
        """

        self.config = AutoConfig.from_pretrained(model_name)
        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.max_length = max_length

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)

        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)) )
        elif hasattr(self.config, 'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = util.import_from_string(self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    def smart_batching_collate(self, batch):
        texts = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text.strip())

            labels.append(example.label)

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def smart_batching_collate_text_only(self, batch):
        texts = [[] for _ in range(len(batch[0]))]

        for example in batch:
            for idx, text in enumerate(example):
                texts[idx].append(text.strip())

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized
    
    def smart_batching_collate_match_all_pairs(self, batch):
        ipt = [[] for _ in range(len(batch[0].texts))]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                ipt[idx].append(text.strip())

            labels.append(example.label)

        nl = [0] * (len(ipt[0]) * len(ipt[0]))

        u = 0
        for i in range(0, len(ipt[0])* len(ipt[0]), len(ipt[0])+1):
          nl[i] = labels[u]
          u += 1

        temp = [[j, i] for j in ipt[0] for i in ipt[1]]

        texts = [[],[]]
                        
        for pair in temp:
          texts[0].append(pair[0])
          texts[1].append(pair[1])

        tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(nl, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def smart_batching_collate_match_triples(self, batch):
        
        #Given 3 pairs: <q1, p1, n1>, <q2, p2, n2>, <q3, p3, n3> (this works for batchsize > 3
        # Create two lists, where Xn are None
        # Q: [q1, q1, q1, q1, q1, q1, q2, q2, q2, q2, q2, q2, q3, q3, q3, q3, q3, q3]
        # P: [p1, n1, X1, X1, X1, X1, p2, n2, X2, X2, X2, X2, p3, n3, X3, X3, X3, X3]
        # L: [ 1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0]
        print("zau")
        print(batch)
        for example in batch:
          print(example)
          break
        ipt = [[],[]]
        labels = []
        for example in batch:
            for idx, text in enumerate(example.texts):
                if idx == 0:
                  ipt[0].append(text.strip())
                  ipt[0].append(text.strip())
                  for _ in range(2 * (len(batch) -1)):
                    ipt[0].append(text.strip())
                else:
                  ipt[1].append(text.strip())
                  if idx == 2:
                    for _ in range(2 * (len(batch) -1)):
                      ipt[1].append(None)
            labels.append(1)
            labels.append(0)
            for _ in range(2 * (len(batch) -1)):
                labels.append(0)

        # Replace the Nones in list P. Pair each query with the other two queries passages.
        # FIXME: This is hardcoded for batchsize = 3.
        if len(batch) == 3:
          ipt[1][2:6] = ipt[1][6:8] + ipt[1][12:14]
          ipt[1][8:12] = ipt[1][0:2] + ipt[1][12:14]
          ipt[1][14:] = ipt[1][0:2] + ipt[1][6:8]
        
        elif len(batch) == 4:
          ipt[1][2:8] = ipt[1][8:10] + ipt[1][16:18] + ipt[1][24:26]
          ipt[1][10:16] = ipt[1][0:2] + ipt[1][16:18] + ipt[1][24:26]
          ipt[1][18:24] = ipt[1][0:2] + ipt[1][8:10] + ipt[1][24:26]
          ipt[1][26:32] = ipt[1][0:2] + ipt[1][8:10] + ipt[1][16:18]


        tokenized = self.tokenizer(*ipt, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def smart_batching_collate_match_triples_weighted(self, batch):
        
        #Given 3 pairs: <q1, p1, n1>, <q2, p2, n2>, <q3, p3, n3> (this works for batchsize > 3
        # Create two lists, where Xn are None
        # Q: [q1, q1, q1, q1, q1, q1, q2, q2, q2, q2, q2, q2, q3, q3, q3, q3, q3, q3]
        # P: [p1, n1, X1, X1, X1, X1, p2, n2, X2, X2, X2, X2, p3, n3, X3, X3, X3, X3]
        # L: [ 1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0]

        ipt = [[],[]]
        labels = []
        weights = []
        for example in batch:
            for idx, text in enumerate(example.texts):
                if idx == 0:
                  ipt[0].append(text.strip())
                  ipt[0].append(text.strip())
                  for _ in range(2 * (len(batch) -1)):
                    ipt[0].append(text.strip())
                else:
                  ipt[1].append(text.strip())
                  if idx == 2:
                    for _ in range(2 * (len(batch) -1)):
                      ipt[1].append(None)
            labels.append(1)
            weights.append(example.label)
            labels.append(0)
            weights.append(example.label)
            for _ in range(2 * (len(batch) -1)):
                labels.append(0)
                weights.append(example.label)

        # Replace the Nones in list P. Pair each query with the other two queries passages.
        # FIXME: This is hardcoded for batchsize = 3.
        if len(batch) == 3:
          ipt[1][2:6] = ipt[1][6:8] + ipt[1][12:14]
          ipt[1][8:12] = ipt[1][0:2] + ipt[1][12:14]
          ipt[1][14:] = ipt[1][0:2] + ipt[1][6:8]
        
        elif len(batch) == 4:
          ipt[1][2:8] = ipt[1][8:10] + ipt[1][16:18] + ipt[1][24:26]
          ipt[1][10:16] = ipt[1][0:2] + ipt[1][16:18] + ipt[1][24:26]
          ipt[1][18:24] = ipt[1][0:2] + ipt[1][8:10] + ipt[1][24:26]
          ipt[1][26:32] = ipt[1][0:2] + ipt[1][8:10] + ipt[1][16:18]


        tokenized = self.tokenizer(*ipt, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, (labels, torch.tensor(weights, dtype=torch.float, device=self._target_device))

    def smart_batching_collate_match_quadruples(self, batch):
        
        #Given 3 pairs: <q1, p1, n11, n12>, <q2, p2, n21, n22>, <q3, p3, n31, n32> (this works for batchsize > 3
        # Create two lists, where Xn are None
        # Q: [q1,  q1,  q1, q1, q1, q1, q1, q1, q1, q2,  q2,  q2, q2, q2, q2, q2, q2, q2, ... 
        # P: [p1, n11, n12, X1, X1, X1, X1, X1, X1, p2, n21, n22, X2, X2, X2, X2, X2, X2, ... 
        # L: [ 1,   0,   0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0]
        ipt = [[],[]]
        labels = []
        for example in batch:
            for idx, text in enumerate(example.texts):
                if idx == 0:
                  ipt[0].append(text.strip())
                  ipt[0].append(text.strip())
                  ipt[0].append(text.strip())
                  for _ in range(3 * (len(batch) -1)):
                    ipt[0].append(text.strip())
                else:
                  ipt[1].append(text.strip())
                  if idx == 3:
                    for _ in range(3 * (len(batch) -1)):
                      ipt[1].append(None)
            labels.append(1)
            labels.append(0)
            labels.append(0)
            for _ in range(3 * (len(batch) -1)):
                labels.append(0)

        # Replace the Nones in list P. Pair each query with the other two queries passages.
        # FIXME: This is hardcoded for batchsize = 3.
        if len(batch) == 3:
          ipt[1][3:9] = ipt[1][9:12] + ipt[1][18:21]
          ipt[1][12:18] = ipt[1][0:3] + ipt[1][18:21]
          ipt[1][21:] = ipt[1][0:3] + ipt[1][9:12]


        tokenized = self.tokenizer(*ipt, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
        labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self._target_device)

        for name in tokenized:
            tokenized[name] = tokenized[name].to(self._target_device)

        return tokenized, labels

    def fit(self,
            train_dataloader: DataLoader,
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = transformers.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            best_eval_output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            batch_wise_easy_negative_pairing: str = None,
            gradient_accumulation_steps: int = 1,
            ):
        """
        Train the model with the given training objective
        Each training objective is sampled in turn for one batch.
        We sample only as many batches from each objective as there are in the smallest one
        to make sure of equal training with each dataset.
        :param train_dataloader: DataLoader with training InputExamples
        :param evaluator: An evaluator (sentence_transformers.evaluation) evaluates the model performance during training on held-out dev data. It is used to determine the best model that is saved to disc.
        :param epochs: Number of epochs for training
        :param loss_fct: Which loss function to use for training. If None, will use nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()
        :param activation_fct: Activation function applied on top of logits output of model.
        :param scheduler: Learning rate scheduler. Available schedulers: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        :param warmup_steps: Behavior depends on the scheduler. For WarmupLinear (default), the learning rate is increased from o up to the maximal learning rate. After these many training steps, the learning rate is decreased linearly back to zero.
        :param optimizer_class: Optimizer
        :param optimizer_params: Optimizer parameters
        :param weight_decay: Weight decay for model parameters
        :param evaluation_steps: If > 0, evaluate the model using evaluator after each number of training steps
        :param output_path: Storage path for the model and evaluation files
        :param save_best_model: If true, the best model (according to evaluator) is stored at output_path
        :param max_grad_norm: Used for gradient normalization.
        :param use_amp: Use Automatic Mixed Precision (AMP). Only for Pytorch >= 1.6.0
        :param callback: Callback function that is invoked after each evaluation.
                It must accept the following three parameters in this order:
                `score`, `epoch`, `steps`
        """
        if batch_wise_easy_negative_pairing == 'pairs':
            train_dataloader.collate_fn = self.smart_batching_collate_match_all_pairs
        elif batch_wise_easy_negative_pairing == 'triples':
            train_dataloader.collate_fn = self.smart_batching_collate_match_triples
        elif batch_wise_easy_negative_pairing == 'triples_weighted':
            train_dataloader.collate_fn = self.smart_batching_collate_match_triples_weighted
        elif batch_wise_easy_negative_pairing == 'quadruples':
            train_dataloader.collate_fn = self.smart_batching_collate_match_quadruples
        else:
          train_dataloader.collate_fn = self.smart_batching_collate

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model.to(self._target_device)

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)
        
        if best_eval_output_path is not None:
            os.makedirs(best_eval_output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) // gradient_accumulation_steps * epochs)
        logger.info("Steps (/GA STEPS): {}".format(num_train_steps))

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)

        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            loss_fct = nn.BCEWithLogitsLoss() if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        has_weighted_loss = False

        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch"):
            training_steps = 0
            self.model.zero_grad()
            self.model.train()

            for features, labels in tqdm(train_dataloader, desc="Iteration", smoothing=0.05):
              # if bugs, remove from here \/
              if isinstance(labels, tuple):
                has_weighted_loss = True
                ws = labels[1]
                labels = labels[0]
                # to here /\
                if use_amp:
                    with autocast():
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                            
                        # and ignore the conditional (i.e, remove the one with ws).
                        if has_weighted_loss:
                          loss_value = loss_fct(logits, labels, ws)
                        else:
                          loss_value = loss_fct(logits, labels)
                        loss_value = loss_value / gradient_accumulation_steps

                    scale_before_step = scaler.get_scale()
                    scaler.scale(loss_value).backward()

                    # https://stackoverflow.com/questions/65842691/final-step-of-pytorch-gradient-accumulation-for-small-datasets
                    # https://discuss.pytorch.org/t/gradient-accumulation-and-scheduler/69077

                    if (training_steps+1) % gradient_accumulation_steps == 0:
                      scaler.unscale_(optimizer)
                      torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                      scaler.step(optimizer)
                      scaler.update()

                      skip_scheduler = scaler.get_scale() != scale_before_step

                      optimizer.zero_grad()

                      if not skip_scheduler:
                        scheduler.step()

                    training_steps += 1

                    
                else:
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                        logits = logits.view(-1)
                    loss_value = loss_fct(logits, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                      scheduler.step()

                    training_steps += 1

                if evaluator is not None and evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    self._eval_during_training(evaluator, best_eval_output_path, save_best_model, epoch, training_steps, callback)

                    self.model.zero_grad()
                    self.model.train()

            if evaluator is not None:
                self._eval_during_training(evaluator, best_eval_output_path, save_best_model, epoch, -1, callback)



    def predict(self, sentences: List[List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               num_workers: int = 0,
               activation_fct = None,
               apply_softmax = False,
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False
               ):
        """
        Performs predicts with the CrossEncoder on the given sentence pairs.
        :param sentences: A list of sentence pairs [[Sent1, Sent2], [Sent3, Sent4]]
        :param batch_size: Batch size for encoding
        :param show_progress_bar: Output progress bar
        :param num_workers: Number of workers for tokenization
        :param activation_fct: Activation function applied on the logits output of the CrossEncoder. If None, nn.Sigmoid() will be used if num_labels=1, else nn.Identity
        :param convert_to_numpy: Convert the output to a numpy matrix.
        :param apply_softmax: If there are more than 2 dimens ions and apply_softmax=True, applies softmax on the logits output
        :param convert_to_tensor:  Conver the output to a tensor.
        :return: Predictions for the passed sentence pairs
        """
        input_was_string = False
        if isinstance(sentences[0], str):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        inp_dataloader = DataLoader(sentences, batch_size=batch_size, collate_fn=self.smart_batching_collate_text_only, num_workers=num_workers, shuffle=False)

        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        iterator = inp_dataloader
        if show_progress_bar:
            iterator = tqdm(inp_dataloader, desc="Batches")

        if activation_fct is None:
            activation_fct = self.default_activation_function

        pred_scores = []
        self.model.eval()
        self.model.to(self._target_device)
        with torch.no_grad():
            for features in iterator:
                model_predictions = self.model(**features, return_dict=True)
                logits = activation_fct(model_predictions.logits)

                if apply_softmax and len(logits[0]) > 1:
                    logits = torch.nn.functional.softmax(logits, dim=1)
                pred_scores.extend(logits)

        if self.config.num_labels == 1:
            pred_scores = [score[0] for score in pred_scores]

        if convert_to_tensor:
            pred_scores = torch.stack(pred_scores)
        elif convert_to_numpy:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])

        if input_was_string:
            pred_scores = pred_scores[0]

        return pred_scores


    def _eval_during_training(self, evaluator, output_path, save_best_model, epoch, steps, callback):
        """Runs evaluation during the training"""
        if evaluator is not None:
            score = evaluator(self, output_path=output_path, epoch=epoch, steps=steps)
            if callback is not None:
                callback(score, epoch, steps)
            if score > self.best_score:
                self.best_score = score
                if save_best_model:
                    self.save(output_path)

    def save(self, path):
        """
        Saves all model and tokenizer to path
        """
        if path is None:
            return

        logger.info("Save model to {}".format(path))
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def save_pretrained(self, path):
        """
        Same function as save
        """
        return self.save(path)