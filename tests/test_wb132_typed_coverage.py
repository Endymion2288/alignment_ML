"""Regression for measurementsOnTrack versus linked outlier TSOS coverage."""
import copy
import unittest
from alignment.wb132_typed_coverage import typed_maps


def state(identifier,flag):
    flags=[False]*12;flags[flag]=True
    return {'type_flags':flags,'measurement':{'cluster_id':identifier,'rot_identifier':identifier,
            'selected_prd':True,'prd_link_resolved':True,'prd_link_valid':True,'selected_prd_pointer_equal':True}}


class TestTypedCoverage(unittest.TestCase):
    def setUp(self):
        self.accepted=state(10469004361915695104,0)
        self.outlier=state(10469004361915695105,5)

    def test_extra_linked_outlier_preserved_without_changing_accepted_membership(self):
        raw=[self.accepted,self.outlier];before=copy.deepcopy(raw)
        all_links,accepted=typed_maps(raw)
        self.assertEqual(set(all_links),{10469004361915695104,10469004361915695105})
        self.assertEqual(set(accepted),{10469004361915695104})
        self.assertEqual(raw,before)

    def test_measurement_bit_does_not_override_outlier_exclusion(self):
        self.outlier['type_flags'][0]=True
        self.assertEqual(len(typed_maps([self.accepted,self.outlier])[1]),1)

    def test_relabelling_outlier_changes_membership_and_is_detectable(self):
        self.outlier['type_flags'][5]=False;self.outlier['type_flags'][0]=True
        self.assertNotEqual(set(typed_maps([self.accepted,self.outlier])[1]),{10469004361915695104})

    def test_duplicate_prd_rejected_even_if_one_state_is_excluded(self):
        self.outlier['measurement']['cluster_id']=self.accepted['measurement']['cluster_id']
        self.outlier['measurement']['rot_identifier']=self.accepted['measurement']['cluster_id']
        with self.assertRaisesRegex(ValueError,'duplicate'):typed_maps([self.accepted,self.outlier])


if __name__=='__main__':unittest.main()
