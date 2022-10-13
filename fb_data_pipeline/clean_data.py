import pandas as pd
import string
import nltk
from cleantext import clean
import os
from PIL import Image


class CleanTabular():

    def __init__(self, tabular_file_path, index_column:bool=True):
        self.df = pd.read_csv(tabular_file_path, lineterminator='\n')
        if index_column:
            self.df = self.df.drop(self.df.columns[0], axis=1)

    
    def clean_to_general_category(self, series:str='category'):
        self.df[series] = self.df[series].apply(lambda x: self.__remove_lower_cat(x))


    def clean_to_county(self, series:str):
        self.df[series] = self.df[series].apply(lambda x: self.__remove_city(x))


    def clean_to_category_type(self, series:str, ohe:bool = True):
        self.df[series] = self.df[series].astype('category')
        d = dict(enumerate(self.df[series].cat.categories))
        self.df[series] = self.df[series].cat.codes
        

        if ohe:
            dummies = pd.get_dummies(self.df[series])
            self.df = pd.concat([self.df, dummies], axis = 1)
            self.df = self.df.drop(columns = [series])

        return d

    def clean_price_to_float(self, series:str, symbol_to_remove:str='£', comma_separated:bool=True):
        self.df[series]= self.df[series].str.strip(symbol_to_remove)

        if comma_separated:
            self.df[series] = self.df[series].str.replace(',', "")

        self.df[series] = self.df[series].astype('float64')
        
    def __remove_punctuations(self, series:str):
        for i in string.punctuation:
            self.df[series]= self.df[series].str.replace(i, "", regex=True)

    def __remove_numbers(self, series:str):
        number_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
        for i in number_list:
            self.df[series]= self.df[series].str.replace(i, "", regex=True)

    def __lower(self, series:str):
        self.df[series] = self.df[series].str.lower()

    def __tokenize(self, series:str):
        self.df[series] = self.df[series].apply(lambda x: nltk.word_tokenize(x))

    def __remove_stopwords(self, series:str):
        stopword = nltk.corpus.stopwords.words('english')
        self.df[series] = self.df[series].apply(lambda x: [word for word in x if word not in stopword])

    def __remove_emoji(self, series:str):
        self.df[series] = self.df[series].apply(lambda x: clean(x, no_emoji=True))

    def __remove_meaningless_char(self, series:str):
        meaningless_char = ["'", "'s", 'x', '*']
        self.df[series] = self.df[series].apply(lambda x: [text for text in x if text not in meaningless_char])

    def __remove_short_str(self, series:str):
        self.df[series] = self.df[series].apply(lambda x: [text for text in x if len(text) > 2])

    def __lemmatize_text(self, series:str):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        self.df[series] = self.df[series].apply(lambda x: [lemmatizer.lemmatize(text) for text in x])
    
    def __stem_text(self, series:str):
        stemmer = nltk.stem.PorterStemmer()
        self.df[series] = self.df[series].apply(lambda x: [stemmer.stem(text) for text in x])

    def __remove_city(self, txt:str):
        if ',' in txt:
            index = txt.index(',')
            return txt[index+2:]
        else:
            return txt
        pass

    def __remove_lower_cat(self, txt:str):
        if '/' in txt:
            index = txt.index('/')
            return txt[:index]
        else:
            return txt

    def clean_text(self, series:str,  lemmatize_stem:bool=True):
        self.__remove_punctuations(series)
        self.__remove_numbers(series)
        self.__remove_emoji(series)
        self.__lower(series)
        self.__tokenize(series)
        self.__remove_stopwords(series)
        self.__remove_meaningless_char(series)
        self.__remove_short_str(series)
        if lemmatize_stem:
            self.__lemmatize_text(series)
        else:
            self.__stem_text(series)
    
    def slice_df(self, series_to_keep:list):
        for col in list(self.df.columns):
            if col not in series_to_keep:
                self.df = self.df.drop(columns=col)


class CleanImage():
    def __init__(self, img_folder_path:str):
        self.img_folder_path = img_folder_path
        self.img_file_list = self.__get_image_path()

    def __get_image_path(self):
        file_list = []
        for i in os.listdir(self.img_folder_path):
            if i[0] != ".":
                file_list.append(i)
        return file_list

    def __create_img_instance(self, img_file):
        img_path = os.path.join(self.img_folder_path, img_file)
        return Image.open(img_path)

    def __get_image_size(self, image:Image):
        return image.size

    def __crop_to_square(self, image:Image, image_size:tuple):
        min_length = min(image_size)
        max_length = max(image_size)
        index_min = image_size.index(min_length)

        if index_min == 1:
            x1 = (max_length-min_length)/2
            x2 = x1 + min_length
            y1 = 0
            y2 = min_length
        else:
            x1 = 0
            x2 = min_length
            y1 = (max_length-min_length)/2
            y2 = y1+min_length

        box = (x1, y1, x2, y2)
        return image.crop(box)

    def __square_padding(self, image:Image, image_size:tuple):
        max_length = max(image_size)
        min_length = min(image_size)
        index_min = image_size.index(min_length)
        pad = Image.new("RGB", (max_length, max_length))
        if index_min == 0:
            x1 = (max_length-min_length)/2
            x2 = 0
        else:
            x1 = 0
            x2 = (max_length-min_length)/2

        pad.paste(image, (int(x1), int(x2)))
        return pad

    def __resize_image(self, image:Image, resized_pixel):
        return image.resize(resized_pixel, Image.Resampling.LANCZOS)

    def __channel_standardisation(self, image:Image, mode):
        return image.convert(mode)

    def process_images(self, crop_img:bool = False, pad_img:bool = True, resized_pixel:tuple = (25,25), greyscale:bool = False, mode:str = 'L'):

        for img in self.img_file_list:
            img_instance = self.__create_img_instance(img)
            img_size = self.__get_image_size(img_instance)

            if img_size[0] != img_size[1]:
                if crop_img:
                    img_instance = self.__crop_to_square(img_instance, img_size)

                if pad_img:
                    img_instance = self.__square_padding(img_instance, img_size)
        
            img_instance = self.__resize_image(img_instance, resized_pixel)

            img_instance = self.__channel_standardisation(img_instance, mode)
            parent_directory = os.path.dirname(self.img_folder_path)
            parent_directory2 = os.path.dirname(parent_directory)
            cleaned_image_path = os.path.join(parent_directory2, "cleaned_image_data", img)
            img_instance.save(cleaned_image_path)
        return os.path.join(parent_directory2, "cleaned_image_data")



if __name__ == '__main__':
    df = CleanTabular('data/Products.csv')
    # print(df.df.info())
    df.clean_to_general_category()

    print(df.df['category'])

    # image=CleanImage("test")
    # image.process_images()
    # insstance = image.create_img_instance(image.img_file_list[0])
    # size = image.get_image_size(insstance)
    # cropped_img = image.crop_to_square(insstance, size)
    # size = image.get_image_size(cropped_img)
    # cropped_img.save('hi.jpg')

        







