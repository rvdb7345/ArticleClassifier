import pandas as pd
from gensim.parsing.preprocessing import remove_stopwords


class DataPreprocessor():
    """Class for preprocessing the extracted csvs to a workable format."""

    def __init__(self):
        pass

    def standard_text_clean_up(self, text_col: pd.Series) -> pd.Series:
        """
        Do standard text cleaning on a general text column. \
        This will:
            - remove all interpunction except white spaces, \
            - lower all text.

        Args:
            text_col (pd.Series): a pandas series containing strings

        Returns:
            A cleaned column of text
        """
        cleaned_text_col = text_col. \
            dropna(). \
            str.lower(). \
            str.replace('[^|\w\s]', '', regex=True). \
            str.replace(r'\r+|\n+|\t+', '', regex=True). \
            str.replace(r' +', ' ', regex=True)

        return cleaned_text_col

    def clean_abstract_data(self, abstracts: pd.Series) -> pd.Series:
        """
        Clean the abstract data.

        Args:
            abstracts (pd.Series): A pandas series containing all the abstracts

        Returns:
            Clean column of all abstracts.
        """
        cleaned_abstracts = self.standard_text_clean_up(abstracts)
        cleaned_abstracts = cleaned_abstracts.apply(lambda x: remove_stopwords(x))

        return cleaned_abstracts

    def clean_keyword_data(self, keywords: pd.Series) -> pd.Series:
        """
        Clean the keyword data.

        Args:
            keywords (pd.Series): A pandas series containing all the keywords

        Returns:
            Clean column of all keywords.
        """
        cleaned_keywords = self.standard_text_clean_up(keywords)
        cleaned_keywords = cleaned_keywords.str.split('|')

        return cleaned_keywords
