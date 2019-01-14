import re

def atoi(text):
    try:
        int(text)
        return int(text)
    except:
        return text

def naturalKeys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(-*\d+)', text)]