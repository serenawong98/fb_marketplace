from torchtext.datasets import WikiText103, IMDB
from torchtext.data import to_map_style_dataset
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import build_vocab_from_iterator
from functools import partial
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import json
import torch
import os
import clean_data

MIN_FREQUENCY = 50
CBOW_N_WORDS = 4
MAX_SEQUENCE_LENGTH = 250
EMBED_DIMENSION = 300
EMBED_MAX_NORM = 1
EPOCH = 10

class FbItemDescript(Dataset):
    def __init__(self, dataframe):
        super().__init__()
        self.data = dataframe

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        features = data_row[1]
        labels = torch.tensor(data_row[0])
        return(features, labels)

    def __len__(self):
        return len(self.data)

def get_data_iterator(ds_type,data_dir):
    data_iter = WikiText103(root=data_dir, split=(ds_type))
    data_iter = to_map_style_dataset(data_iter)
    return data_iter

def get_english_tokenizer():
    tokenizer = get_tokenizer("basic_english", language="en")
    return tokenizer

def yield_tokens(data_iter):
    tokenizer = get_tokenizer('basic_english')
    for text, _ in data_iter:
        yield tokenizer(text)

def build_vocab(data_iter, tokenizer, ds_iter):
    if ds_iter == None:
        token_data_iter = map(tokenizer, data_iter)
    else:
        token_data_iter = yield_tokens(data_iter)
    
    vocab = build_vocab_from_iterator(
        token_data_iter,
        specials=["<unk>"],
        min_freq=MIN_FREQUENCY,
    )
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def collate_cbow(batch, text_pipeline):
    batch_input = []
    batch_output = []
    # get rid of ,_ for dataloaders
    for text, _ in batch:
        text_tokens_ids = text_pipeline(text)

        if len(text_tokens_ids) < CBOW_N_WORDS*2+1:
            continue
        
        if MAX_SEQUENCE_LENGTH:
            text_tokens_ids = text_tokens_ids[:MAX_SEQUENCE_LENGTH]
        
        for idx in range(len(text_tokens_ids)-CBOW_N_WORDS*2):
            token_id_sequence = text_tokens_ids[idx: (idx+CBOW_N_WORDS*2+1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)
    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output

def get_dataloader_and_vocab(ds_type, data_dir, batch_size, shuffle, vocab=None, ds_iter=None):
    if ds_iter == None:
        data_iter = get_data_iterator(ds_type, data_dir)
    else:
        data_iter = ds_iter

    tokenizer = get_english_tokenizer()

    if not vocab:
        vocab = build_vocab(data_iter, tokenizer, ds_iter)

    text_pipeline = lambda x: vocab(tokenizer(x))

    dataloader = DataLoader(
        data_iter,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_cbow, text_pipeline=text_pipeline)
    )
    return dataloader, vocab

class CBOW_MODEL(nn.Module):
    def __init__(self, vocab_size:int):
        super(CBOW_MODEL, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION, 
            max_norm=EMBED_MAX_NORM
        )
        self.linear = nn.Linear(
            in_features=EMBED_DIMENSION,
            out_features=vocab_size
        )
    
    def forward(self, inputs_):
        x = self.embeddings(inputs_)
        x = x.mean(axis = 1)
        x = self.linear(x)
        return x

class Trainer:
    def __init__(
        self, 
        model,
        epochs,
        train_dataloader,
        train_steps,
        val_dataloader,
        val_steps,
        checkpoint_frequency,
        criterion,
        optimizer,
        lr_scheduler,
        device, 
        model_dir,
    ):

        self.model = model
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.train_steps = train_steps
        self.val_dataloader = val_dataloader
        self.val_steps = val_steps
        self.criterion = criterion
        self.optimizer = optimizer
        self.checkpoint_frequency = checkpoint_frequency
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.model_dir = model_dir

        self.loss = {"train": [], "val": []}
        self.model.to(self.device)
        try:
            os.mkdir(self.model_dir)
        except FileExistsError:
            pass

    def train(self):

        writer =  SummaryWriter()
        batch_idx = 0
    
        for epoch in range(self.epochs):
            self._train_epoch(writer, batch_idx)
            self._validate_epoch()
            print(
                "Epoch: {}/{}, Train Loss={:.5f}, Val Loss{:.5f}".format(
                    epoch+1, 
                    self.epochs,
                    self.loss["train"][-1],
                    self.loss["val"][-1],
                )
            )

            self.lr_scheduler.step()

            if self.checkpoint_frequency:
                self._save_checkpoint(epoch)
    
    def _train_epoch(self, writer, batch_idx):
        self.model.train()
        running_loss = []

        for i, batch_data in enumerate(self.train_dataloader, 1):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            writer.add_scalar('loss', loss.item(), batch_idx)

            running_loss.append(loss.item())

            if i == self.train_steps:
                break
            batch_idx += 1
        
        epoch_loss = np.mean(running_loss)
        self.loss["train"].append(epoch_loss)

    def _validate_epoch(self):
        self.model.eval()
        running_loss=[]

        with torch.no_grad():
            for i, batch_data in enumerate(self.val_dataloader, 1):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss.append(loss.item())

                if i == self.val_steps:
                    break

        epoch_loss = np.mean(running_loss)
        self.loss["val"].append(epoch_loss)

    def _save_checkpoint(self,epoch):
        epoch_num = epoch+1
        if epoch_num % self.checkpoint_frequency == 0:
            model_path = "checkpoint_{}.pt".format(str(epoch_num.zfill(3)))
            model_path = os.path.join(self.model_dir, model_path)
            torch.save(self.model, model_path)

    def save_model(self):
        model_path = os.path.join(self.model_dir, "model.pt")
        torch.save(self.model, model_path)

    def save_loss(self):
        loss_path = os.path.join(self.model_dir, "loss.json")
        with open(loss_path, "w") as fp:
            json.dump(self.loss, fp)

def save_vocab(vocab, model_dir: str):
    vocab_path = os.path.join(model_dir, "vocab.pt")
    torch.save(vocab, vocab_path)

def get_lr_scheduler(optimizer, total_epochs: int, verbose: bool = True):

    lr_lambda = lambda epoch: (total_epochs - epoch) / total_epochs
    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=verbose)
    return lr_scheduler





if __name__ == "__main__":

    # filepath = os.path.join(os.getcwd(), 'data', "vocab_text.csv")
    fb_clean = clean_data.CleanTabular(os.path.join(os.getcwd(), 'data/Products.csv'))
    fb_clean.slice_df(series_to_keep = ['product_description', 'category'])
    fb_clean.clean_to_general_category('category')
    fb_clean.clean_to_category_type('category', ohe=False)

    word2vec_dataset = FbItemDescript(fb_clean.df)
    # features, labels =  word2vec_dataset[0]
    # print(len(word2vec_dataset))

    train_set, test_set = torch.utils.data.random_split(word2vec_dataset, [6656, 500])

    train_loader, vocab = get_dataloader_and_vocab(
        ds_type="train",
        data_dir="data/",
        batch_size=96,
        shuffle=True,
        vocab=None,
        ds_iter = train_set,
    )

    val_loader, _ = get_dataloader_and_vocab(
        ds_type="valid",
        data_dir="data/",
        batch_size=96,
        shuffle=True,
        vocab=None,
        ds_iter = test_set,

    )

    vocab_size = len(vocab.get_stoi())
    print(f"Vocabulary size: {vocab_size}")

    model = CBOW_MODEL(vocab_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr = 0.025)
    lr_scheduler = get_lr_scheduler(optimizer, EPOCH, verbose=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_dir = os.path.join(os.getcwd(), "word2vec")

    trainer = Trainer(
        model,
        epochs=EPOCH,
        train_dataloader=train_loader,
        train_steps= None,
        val_dataloader= val_loader,
        val_steps= None,
        criterion=criterion,
        optimizer=optimizer,
        checkpoint_frequency= None,
        lr_scheduler=lr_scheduler,
        device=device,
        model_dir=model_dir,
    )

    trainer.train()
    print("Training Finished")

    trainer.save_model()
    trainer.save_loss()
    save_vocab(vocab, model_dir)
