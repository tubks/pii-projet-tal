import re 

EMAIL_REGEX = r'\b[A-Za-z0-9]+[.-_]*[A-Za-z0-9]+@(?:gmail|hotmail|yahoo)\.[A-Z|a-z]{2,}\b'
PHONE_NUMEBR_REGEX = r'\(\d{1,3}\)\d{3}-\d+(?:x\d+)?|\d{3}\.\d{3}\.\d{1,10}'
ADDRESS_REGEX = r'\b\d+\s+[a-zA-Z\s]+(?:\. \d+)[a-zA-Z\s]+(?:\, [A-Z]{2}\s\d{5,6})\b'
NUMBER_REGEX = r'\d+'


def find_email(text):                              
    return re.findall(re.compile(EMAIL_REGEX), text)


def find_phone_number(text):
    return re.findall(re.compile(PHONE_NUMEBR_REGEX), text)


def find_address(text):
    return re.findall(re.compile(ADDRESS_REGEX), text)


def check_number(token):
    return re.fullmatch(re.compile(NUMBER_REGEX), token)
