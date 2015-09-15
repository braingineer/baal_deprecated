"""
This will be the default interface.  Inherit this. Add generic functions here.
"""

class corpus(object):

    def __init__(self):
        pass

    @staticmethod
    def get_file(fileid):
        """ if corpus is split into files, use this. """
        pass

    @staticmethod
    def make_all():
        """ Iterator for all files or parts of the corpus """
        pass


class corpus_data(object):

    def __init__(self, name, *args, **kwargs):
        self.name = name


class corpus_datum(object):

    def __init__(self, *args, **kwargs):
        pass

    def __str__(self):
        pass

    def __repr__(self):
        pass
