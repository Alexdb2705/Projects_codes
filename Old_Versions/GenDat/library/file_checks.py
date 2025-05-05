

def check_file_arg(exception, filePath):
    """
        check_file_arg:: Exception -> Path -> IO Path | IO Exception
        check whether the file arg is a file and is readable
        exception provides a suitable exception from the caller
    """
    if not (filePath.exists() and filePath.is_file()):                          # check path is readable file
        raise exception                                                         # or raise exception
    return filePath                                                             # return path

def check_dir_arg(exception, filePath):
    """
        check_dir_arg:: Exception -> Path -> IO Path | IO Exception
        check whether the path arg is not a file or existing directory
        exception provides a suitable exception from the caller
    """
    if filePath.exists():                                   # does this path exists
        raise exception                                     # then bail out
    return filePath                                         # else return path nicely