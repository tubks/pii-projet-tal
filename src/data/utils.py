import re 

EMAIL_REGEX = r'([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+'
PHONE_NUMEBR_REGEX = ""


def check_email(text):
    return re.findall(EMAIL_REGEX, text)


def check_phone_number():
    pass