from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Embedding, FeedForward, TimeDistributed

from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import RnnSeq2SeqEncoder, GruSeq2SeqEncoder, LstmSeq2SeqEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import Activation

from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.nn.util import get_text_field_mask
from allennlp.training.metrics import CategoricalAccuracy

## Modèles sans embedding pre-entrainé 
class POS_ZH_Model(Model):
    def __init__(self, vocab: Vocabulary, dim=10, **kwargs):
        super().__init__(vocab, **kwargs)
        self.num_classes = vocab.get_vocab_size("labels")

        self.text_embedder = BasicTextFieldEmbedder( 
            {'tokens': Embedding(dim, vocab=vocab, pretrained_file=None)}
        )
        self.seq2seq = RnnSeq2SeqEncoder(self.text_embedder.get_output_dim(),
                                         self.text_embedder.get_output_dim())

        self.ff = TimeDistributed(
            FeedForward(input_dim=self.seq2seq.get_output_dim(),
                              num_layers=1,
                              hidden_dims=self.num_classes,
                              activations=Activation.by_name("relu")(),
							  dropout=[0.20]
                              )
        )
        self.accuracy = CategoricalAccuracy()

    def forward(self, tokens, labels) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embedding = self.text_embedder(tokens)
        encoded = self.seq2seq(embedding, mask)
        logits = self.ff(encoded)
        loss = sequence_cross_entropy_with_logits(logits, labels, mask)
        self.accuracy(logits, labels, mask)
        return {'loss': loss}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'acc': self.accuracy.get_metric(reset)}

class POS_ZH_ModelGru(Model):
    def __init__(self, vocab: Vocabulary, dim=10, **kwargs):
        super().__init__(vocab, **kwargs)
        self.num_classes = vocab.get_vocab_size("labels")

        self.text_embedder = BasicTextFieldEmbedder( 
            {'tokens': Embedding(dim, vocab=vocab, pretrained_file=None)} 
        )

        self.seq2seq = GruSeq2SeqEncoder(self.text_embedder.get_output_dim(),
                                         self.text_embedder.get_output_dim())

        self.ff = TimeDistributed(
            FeedForward(input_dim=self.seq2seq.get_output_dim(),
                              num_layers=1,
                              hidden_dims=self.num_classes,
                              activations=Activation.by_name("relu")(),
							  dropout=[0.20]
                              )
        )
        self.accuracy = CategoricalAccuracy()

    def forward(self, tokens, labels) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embedding = self.text_embedder(tokens)
        encoded = self.seq2seq(embedding, mask)
        logits = self.ff(encoded)
        loss = sequence_cross_entropy_with_logits(logits, labels, mask)
        self.accuracy(logits, labels, mask)
        return {'loss': loss}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'acc': self.accuracy.get_metric(reset)}

class POS_ZH_ModelLstm(Model):
    def __init__(self, vocab: Vocabulary, dim=10, **kwargs):
        super().__init__(vocab, **kwargs)
        self.num_classes = vocab.get_vocab_size("labels")

        self.text_embedder = BasicTextFieldEmbedder( 
            {'tokens': Embedding(dim, vocab=vocab, pretrained_file=None)}
        )

        self.seq2seq = LstmSeq2SeqEncoder(self.text_embedder.get_output_dim(),
                                        self.text_embedder.get_output_dim())

        self.ff = TimeDistributed(
            FeedForward(input_dim=self.seq2seq.get_output_dim(),
                              num_layers=1,
                              hidden_dims=self.num_classes,
                              activations=Activation.by_name("relu")(),
							  dropout=[0.20]
                              )
        )
        self.accuracy = CategoricalAccuracy()

    def forward(self, tokens, labels) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embedding = self.text_embedder(tokens)
        encoded = self.seq2seq(embedding, mask)
        logits = self.ff(encoded)
        loss = sequence_cross_entropy_with_logits(logits, labels, mask)
        self.accuracy(logits, labels, mask)
        return {'loss': loss}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'acc': self.accuracy.get_metric(reset)}

## Modèle avec embedding pre-entrainé

class POS_ZH_Model_embedding(Model):
    def __init__(self, vocab: Vocabulary, dim=10, **kwargs):
        super().__init__(vocab, **kwargs)
        self.num_classes = vocab.get_vocab_size("labels")

        self.text_embedder = BasicTextFieldEmbedder( {'tokens': Embedding(dim, vocab=vocab, pretrained_file="./data/embedding/character.vec.txt")} )

        self.seq2seq = RnnSeq2SeqEncoder(self.text_embedder.get_output_dim(),
                                         self.text_embedder.get_output_dim())

        self.ff = TimeDistributed(
            FeedForward(input_dim=self.seq2seq.get_output_dim(),
                              num_layers=1,
                              hidden_dims=self.num_classes,
                              activations=Activation.by_name("relu")(),
							  dropout=[0.20]
                              )
        )
        self.accuracy = CategoricalAccuracy()

    def forward(self, tokens, labels) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embedding = self.text_embedder(tokens)
        encoded = self.seq2seq(embedding, mask)
        logits = self.ff(encoded)
        loss = sequence_cross_entropy_with_logits(logits, labels, mask)
        self.accuracy(logits, labels, mask)
        return {'loss': loss}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'acc': self.accuracy.get_metric(reset)}

class POS_ZH_ModelGru_embedding(Model):
    def __init__(self, vocab: Vocabulary, dim=10, **kwargs):
        super().__init__(vocab, **kwargs)
        self.num_classes = vocab.get_vocab_size("labels")

        self.text_embedder = BasicTextFieldEmbedder( {'tokens': Embedding(dim, vocab=vocab, pretrained_file="./data/embedding/character.vec.txt")} )

        self.seq2seq = GruSeq2SeqEncoder(self.text_embedder.get_output_dim(),
                                         self.text_embedder.get_output_dim())

        self.ff = TimeDistributed(
            FeedForward(input_dim=self.seq2seq.get_output_dim(),
                              num_layers=1,
                              hidden_dims=self.num_classes,
                              activations=Activation.by_name("relu")(),
							  dropout=[0.20]
                              )
        )
        self.accuracy = CategoricalAccuracy()

    def forward(self, tokens, labels) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embedding = self.text_embedder(tokens)
        encoded = self.seq2seq(embedding, mask)
        logits = self.ff(encoded)
        loss = sequence_cross_entropy_with_logits(logits, labels, mask)
        self.accuracy(logits, labels, mask)
        return {'loss': loss}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'acc': self.accuracy.get_metric(reset)}

class POS_ZH_ModelLstm_embedding(Model):
    def __init__(self, vocab: Vocabulary, dim=10, **kwargs):
        super().__init__(vocab, **kwargs)
        self.num_classes = vocab.get_vocab_size("labels")

        self.text_embedder = BasicTextFieldEmbedder( {'tokens': Embedding(dim, vocab=vocab, pretrained_file="./data/embedding/character.vec.txt")} )

        self.seq2seq = LstmSeq2SeqEncoder(self.text_embedder.get_output_dim(),
                                        self.text_embedder.get_output_dim())

        self.ff = TimeDistributed(
            FeedForward(input_dim=self.seq2seq.get_output_dim(),
                              num_layers=1,
                              hidden_dims=self.num_classes,
                              activations=Activation.by_name("relu")(),
							  dropout=[0.20]
                              )
        )
        self.accuracy = CategoricalAccuracy()

    def forward(self, tokens, labels) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embedding = self.text_embedder(tokens)
        encoded = self.seq2seq(embedding, mask)
        logits = self.ff(encoded)
        loss = sequence_cross_entropy_with_logits(logits, labels, mask)
        self.accuracy(logits, labels, mask)
        return {'loss': loss}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {'acc': self.accuracy.get_metric(reset)}


if __name__ == '__main__':
    from readers import ZH_POS_Reader
    from pathlib import Path
    data_path = Path("../data/UD_Chinese-GSDSimp")
    reader = ZH_POS_Reader(max_instances=10)
    instances = list(reader.read( data_path / "zh_gsdsimp-ud-train.conllu"))
    vocab = Vocabulary.from_instances(instances)
    model = POS_ZH_Model(vocab)
    model.forward_on_instances(instances)








