from ssac.io import stable_int


def test_stable_int():
    x = stable_int("abc::attempt=0")
    assert x == stable_int("abc::attempt=0")
    assert x != stable_int("abc::attempt=1")
