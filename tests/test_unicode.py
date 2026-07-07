"""Unit tests for Unicode subsets."""

import itertools
import re
import sys

import pytest

from setfield import AllUnicode, SubsetComplement, UnicodeRanges
from setfield.unicode import NUM_UNICODE


ALL_UNICODE = AllUnicode()
EMPTY = UnicodeRanges([])
UNICODE = UnicodeRanges([range(NUM_UNICODE)])
LATIN = UnicodeRanges([range(128)])
PUNCTUATION = UnicodeRanges([range(33, 48), range(58, 65), range(91, 97), range(123, 127)])


def test_all_unicode():
    """Tests the AllUnicode class."""
    chars = ['\x00', 'a', chr(sys.maxunicode)]
    for c in chars:
        assert c in ALL_UNICODE
    for obj in [1, (), 'ab']:
        assert obj not in ALL_UNICODE
    assert len(ALL_UNICODE) == sys.maxunicode + 1
    n = 100
    assert list(itertools.islice(ALL_UNICODE, n)) == list(map(chr, range(n)))
    assert AllUnicode() is AllUnicode()
    assert AllUnicode() is ALL_UNICODE
    assert AllUnicode() == ALL_UNICODE
    assert AllUnicode() != set()
    assert set() != AllUnicode()
    char_set = set(chars)
    assert char_set <= ALL_UNICODE
    assert ALL_UNICODE >= char_set
    assert char_set < ALL_UNICODE
    assert ALL_UNICODE > char_set
    assert str(ALL_UNICODE) == 'AllUnicode()'

def test_unicode_ranges():
    """Tests the UnicodeRanges class."""
    with pytest.raises(ValueError, match='invalid range'):
        _ = UnicodeRanges([range(-10, 10)])
    # empty set
    assert len(EMPTY) == 0
    assert EMPTY < LATIN
    assert LATIN | EMPTY == LATIN
    assert LATIN & EMPTY == EMPTY
    neg_empty = ~EMPTY
    assert type(neg_empty) is SubsetComplement
    assert len(neg_empty) == NUM_UNICODE
    assert neg_empty == UNICODE
    assert ~neg_empty == EMPTY
    # full Unicode set
    assert len(UNICODE) == NUM_UNICODE
    assert LATIN < UNICODE
    assert LATIN | UNICODE == UNICODE
    assert LATIN & UNICODE == LATIN
    neg_unicode = ~UNICODE
    assert type(neg_unicode) is SubsetComplement
    assert len(neg_unicode) == 0
    assert neg_unicode == set()
    assert ~neg_unicode == UNICODE
    # other sets
    assert LATIN.universe is ALL_UNICODE
    assert len(LATIN) == 128
    assert len(PUNCTUATION) == 32
    assert set(LATIN) == LATIN
    assert LATIN == set(LATIN)
    assert set(PUNCTUATION) == PUNCTUATION
    assert PUNCTUATION == set(PUNCTUATION)
    assert PUNCTUATION < LATIN
    assert PUNCTUATION < set(LATIN)
    assert set(PUNCTUATION) < LATIN
    assert LATIN > PUNCTUATION
    assert LATIN == LATIN
    assert LATIN is not UnicodeRanges([range(128)])
    assert LATIN == UnicodeRanges([range(128)])
    assert LATIN != PUNCTUATION

@pytest.mark.parametrize('ranges', [
    EMPTY,
    UNICODE,
    LATIN,
    PUNCTUATION,
])
def test_unicode_ranges_repr(ranges):
    """Tests the repr behavior of UnicodeRanges."""
    assert str(ranges) == repr(ranges)
    assert re.fullmatch(r'UnicodeRanges\(\[.*\]\)', str(ranges))
    assert eval(str(ranges)) == ranges
