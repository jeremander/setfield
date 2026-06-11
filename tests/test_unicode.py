"""Unit tests for Unicode subsets."""

import itertools
import sys

import pytest

from setfield import AllUnicode, SubsetComplement, UnicodeRanges
from setfield.unicode import NUM_UNICODE


ALL_UNICODE = AllUnicode()


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
    empty = UnicodeRanges([])
    unicode = UnicodeRanges([range(NUM_UNICODE)])
    latin = UnicodeRanges([range(128)])
    punctuation = UnicodeRanges([range(33, 48), range(58, 65), range(91, 97), range(123, 127)])
    # empty set
    assert len(empty) == 0
    assert empty < latin
    assert latin | empty == latin
    assert latin & empty == empty
    neg_empty = ~empty
    assert type(neg_empty) is SubsetComplement
    assert len(neg_empty) == NUM_UNICODE
    assert neg_empty == unicode
    assert ~neg_empty == empty
    # full Unicode set
    assert len(unicode) == NUM_UNICODE
    assert latin < unicode
    assert latin | unicode == unicode
    assert latin & unicode == latin
    neg_unicode = ~unicode
    assert type(neg_unicode) is SubsetComplement
    assert len(neg_unicode) == 0
    assert neg_unicode == set()
    assert ~neg_unicode == unicode
    # other sets
    assert latin.universe is ALL_UNICODE
    assert len(latin) == 128
    assert len(punctuation) == 32
    assert set(latin) == latin
    assert latin == set(latin)
    assert set(punctuation) == punctuation
    assert punctuation == set(punctuation)
    assert punctuation < latin
    assert punctuation < set(latin)
    assert set(punctuation) < latin
    assert latin > punctuation
    assert latin == latin
    assert latin is not UnicodeRanges([range(128)])
    assert latin == UnicodeRanges([range(128)])
    assert latin != punctuation
