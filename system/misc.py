import os
import errno
import re


def makedirs(dir):
    if os.path.exists(dir) is False:
        try:
            os.makedirs(dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
