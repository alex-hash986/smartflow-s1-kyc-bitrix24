import re


def _digits(number: str, length: int) -> bool:
    return re.fullmatch(rf"[0-9]{{{length}}}", number) is not None


def valid_inn(number: str) -> bool:
    weights: tuple[tuple[int, ...], ...]
    if _digits(number, 10):
        weights = ((2, 4, 10, 3, 5, 9, 4, 6, 8),)
    elif _digits(number, 12):
        weights = ((7, 2, 4, 10, 3, 5, 9, 4, 6, 8), (3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8))
    else:
        return False
    digits = [int(digit) for digit in number]
    return all(
        sum(digit * weight for digit, weight in zip(digits, row, strict=False)) % 11 % 10
        == digits[len(row)]
        for row in weights
    )


def valid_ogrn(number: str) -> bool:
    return _digits(number, 13) and int(number[:-1]) % 11 % 10 == int(number[-1])


def valid_ogrnip(number: str) -> bool:
    return _digits(number, 15) and int(number[:-1]) % 13 % 10 == int(number[-1])


def valid_kpp(number: str) -> bool:
    return _digits(number, 9)
