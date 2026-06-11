import pytest

from setfield.ranges import _ranges_complement, _ranges_intersection, _ranges_union, indices_to_minimal_ranges


class TestRanges:

    @pytest.mark.parametrize(['indices', 'ranges'], [
        (
            [],
            [],
        ),
        (
            [0],
            [range(0, 1)],
        ),
        (
            [0, 1, 2],
            [range(0, 3)],
        ),
        (
            [10, 11, 12],
            [range(10, 13)],
        ),
        (
            [0, 2, 3],
            [range(0, 1), range(2, 4)],
        ),
        (
            [3, 0, 2],
            [range(0, 1), range(2, 4)],
        ),
    ])
    def test_indices_to_minimal_ranges(self, indices, ranges):
        assert indices_to_minimal_ranges(indices) == ranges

    @pytest.mark.parametrize(['universe', 'ranges_seq', 'intersection'], [
        (
            range(10),
            [],
            [range(10)],
        ),
        (
            range(10),
            [[]],
            [],
        ),
        (
            range(10),
            [[range(0, 0)]],
            [],
        ),
        (
            range(10),
            [[range(10)]],
            [range(10)],
        ),
        (
            range(10),
            [[range(20)]],
            [range(10)],
        ),
        (
            range(10),
            [[range(-20, 20)]],
            [range(10)],
        ),
        (
            range(10),
            [[range(-20, 5)]],
            [range(5)],
        ),
        (
            range(10),
            [[range(0, 3), range(6, 9)]],
            [range(0, 3), range(6, 9)],
        ),
        (
            range(10),
            [[range(0, 3), range(2, 5)]],
            [range(5)],
        ),
        (
            range(10),
            [[range(0, 3)], [range(6, 9)]],
            [],
        ),
        (
            range(10),
            [[range(0, 3)], [range(2, 5)]],
            [range(2, 3)],
        ),
        (
            range(10),
            [[range(2, 5)], [range(0, 3)]],
            [range(2, 3)],
        ),
    ])
    def test_ranges_intersection(self, universe, ranges_seq, intersection):
        assert _ranges_intersection(universe, ranges_seq) == intersection

    @pytest.mark.parametrize(['ranges_seq', 'union'], [
        (
            [],
            [],
        ),
        (
            [[]],
            [],
        ),
        (
            [[range(0, 10)]],
            [range(0, 10)],
        ),
        (
            [[range(0, 10), range(20, 30)]],
            [range(0, 10), range(20, 30)],
        ),
        (
            [[range(0, 10), range(5, 15)]],
            [range(0, 15)],
        ),
        (
            [[range(0, 10)], [range(5, 15)]],
            [range(0, 15)],
        ),
        (
            [[range(0, 10), range(20, 30)], [range(5, 25)]],
            [range(0, 30)],
        ),
        # empty range
        (
            [[range(10, 10)]],
            [],
        ),
        (
            [[range(0, 10), range(5, 5)]],
            [range(0, 10)],
        ),
        (
            [[range(0, 10), range(10, 10)]],
            [range(0, 10)],
        ),
    ])
    def test_ranges_union(self, ranges_seq, union):
        assert _ranges_union(ranges_seq) == union

    @pytest.mark.parametrize(['universe', 'ranges', 'complement'], [
        (
            range(100),
            [],
            [range(0, 100)],
        ),
        (
            range(100),
            [range(0, 0)],
            [range(0, 100)],
        ),
        (
            range(100),
            [range(0, 1)],
            [range(1, 100)],
        ),
        (
            range(100),
            [range(0, 10)],
            [range(10, 100)],
        ),
        (
            range(100),
            [range(10, 20)],
            [range(0, 10), range(20, 100)],
        ),
        (
            range(100),
            [range(0, 10), range(20, 30)],
            [range(10, 20), range(30, 100)],
        ),
        (
            range(100),
            [range(0, 10), range(10, 20)],
            [range(20, 100)],
        ),
        (
            range(100),
            [range(0, 10), range(5, 15)],
            [range(15, 100)],
        ),
        (
            range(100),
            [range(0, 10), range(5, 10)],
            [range(10, 100)],
        ),
        (
            range(100),
            [range(0, 10), range(4, 6)],
            [range(10, 100)],
        ),
    ])
    def test_ranges_complement(self, universe, ranges, complement):
        assert _ranges_complement(universe, ranges) == complement
