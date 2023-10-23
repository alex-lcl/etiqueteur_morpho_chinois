from typing import Iterable, List

from pathlib import Path

from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import TextField, SequenceLabelField, LabelField
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.token_indexers import SingleIdTokenIndexer

class ZH_POS_Reader(DatasetReader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.indexer = SingleIdTokenIndexer()


    def _read(self, file_path) -> Iterable[Instance]:
        with open(file_path, "r") as f:
            labels=[]
            text=""
            for line in f:
                if "#" in line:
                    if labels != [] and "text =" in line:
                        yield self.text_to_instance(labels, text)
                        text = ""
                        labels=[]
                    else:
                        continue
                elif "\t" in line:
                    label = line.split("\t")[3]
                    tokens = line.split("\t")[1]
                    text+= tokens
                    count = 0
                    for character in tokens:
                        if count == 0:
                            label_c = "B-"+label
                        else:
                            label_c = "I-"+label
                        count+=1
                        labels.append(label_c)
            yield self.text_to_instance(labels, text)


    def text_to_instance(self, labels: List[str], text: str) -> Instance:
        tokens = [Token(c) for c in text]
        text_field = TextField(tokens, token_indexers={'tokens': self.indexer})
        label_field = SequenceLabelField(labels, text_field)
        return Instance({'labels': label_field, 'tokens': text_field})

if __name__ == "__main__":
    reader = ZH_POS_Reader()
    instances = list(reader.read("/media/sf_M1/S2/MethodeApprentissageAuto/_projet/data/UD_Chinese-GSDSimp/zh_gsdsimp-ud-test.conllu"))
    print(len(instances))
    for i in instances:
        print(i)