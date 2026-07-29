"""
Filename-sequence helpers for numbered frame/backup files.
"""

import os
import re


def get_last_file(directory, pattern):
    """
    Return (filename, index) of the highest-indexed file in `directory`
    matching `pattern` (a regex with a named `index` group), or (None, None).
    """

    def key(f):
        index = re.match(pattern, f).group("index")
        return 0 if index == "" else int(index)

    files = [f for f in os.listdir(directory) if re.match(pattern, f)]
    if len(files) == 0:
        return None, None
    files.sort(key=key)
    index = key(files[-1])
    return files[-1], index


def get_next_file(directory, pattern, templates):
    """
    Given a directory, a file pattern, and a list of templates,
    return the next file name and index that matches the pattern.

    If no files match the pattern, return the first template and 0.

    :param directory: The directory where the files are located
    :param pattern: The pattern to match files against (regex with named
        `pre`/`index`/`post` groups)
    :param templates: A list of file names that are used to create the new file
    :return: The next file name and the next index.
    """

    files = [f for f in os.listdir(directory) if re.match(pattern, f)]
    if len(files) == 0:
        return templates[0], 0

    def key(f):
        index = re.match(pattern, f).group("index")
        return 0 if index == "" else int(index)

    files.sort(key=key)
    n = len(templates) - 1
    for i, f in enumerate(files):
        index = key(f)
        if i != index:
            return (
                (templates[0], 0)
                if i == 0
                else (
                    re.sub(
                        pattern,
                        lambda m, i=i: f"{m.group('pre')}{i}{m.group('post')}",
                        templates[min(i, n)],
                    ),
                    i,
                )
            )
    return (
        re.sub(
            pattern,
            lambda m, i=i: f"{m.group('pre')}{i + 1}{m.group('post')}",
            templates[min(i, n)],
        ),
        i + 1,
    )
