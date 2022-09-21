import flair
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer
import torch
from torch.optim.lr_scheduler import OneCycleLR
from argparse import ArgumentParser


if __name__ == "__main__":
    # All arguments that can be passed
    parser = ArgumentParser()
    parser.add_argument("-c", "--cuda", type=int, default=0, help="CUDA device")  # which cuda device to use

    # Parse experimental arguments
    args = parser.parse_args()

    # use cuda device as passed
    flair.device = None
    if torch.cuda.is_available():
        flair.device = torch.device(f'cuda:{str(args.cuda)}')
    else:
        flair.device = torch.device('cpu')

    # define columns
    columns = {0: 'text', 1: 'ner'}

    # this is the folder in which train and test files reside
    data_folder = '/bask/homes/s/sopk4161/ner/data_entity'

    # 1. init a corpus using column format, data folder and the names of the train, dev and test files
    corpus: Corpus = ColumnCorpus(data_folder, columns,
                                  train_file='train.txt',
                                  dev_file='dev.txt',
                                  test_file='test.txt')

    # 2. what label do we want to predict?
    label_type = 'ner'

    # 3. make the label dictionary from the corpus
    label_dict = corpus.make_label_dictionary(label_type=label_type)

    # 4. initialize fine-tuneable transformer embeddings WITH document context
    embeddings = TransformerWordEmbeddings(model='dslim/bert-base-NER',
                                           layers="-1",
                                           subtoken_pooling="first",
                                           fine_tune=True,
                                           use_context=True,
                                           layer_mean=False
                                           )

    # 5. initialize sequence tagger
    tagger = SequenceTagger(hidden_size=256,
                            embeddings=embeddings,
                            tag_dictionary=label_dict,
                            tag_type='ner',
                            use_crf=True,
                            use_rnn=False,
                            reproject_embeddings=False,
                            )

    # 6. initialize trainer
    trainer = ModelTrainer(tagger, corpus)

    # 7. run fine-tuning
    trainer.fine_tune('resources/taggers/ner-entity-flert',
                      learning_rate=5.0e-6,
                      mini_batch_size=4,
                      max_epochs=20,
                      scheduler=OneCycleLR
                      )