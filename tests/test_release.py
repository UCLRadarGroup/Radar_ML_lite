import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'hackathon'))
import csv
import tempfile
from pathlib import Path
import unittest
from score_submission import score


class ScoreTests(unittest.TestCase):
    def test_perfect_wrong_and_invalid_submissions(self):
        with tempfile.TemporaryDirectory() as directory:
            key, submission = Path(directory)/'key.csv', Path(directory)/'submission.csv'
            key.write_text('sample_id,split,label,requested_snr_db\na,test,cw,0\nb,test,fsk,0\nc,test,cw,-5\nd,test,fsk,-5\n')
            submission.write_text('sample_id,predicted_label\na,cw\nb,fsk\nc,cw\nd,fsk\n')
            self.assertEqual(score(key, submission)['mean_balanced_accuracy'], 1)
            submission.write_text('sample_id,predicted_label\na,fsk\nb,cw\nc,fsk\nd,cw\n')
            self.assertEqual(score(key, submission)['mean_balanced_accuracy'], 0)
            for text in ['a,cw\na,cw\n', 'a,cw\n', 'a,unknown\n', 'z,cw\nb,fsk\nc,cw\nd,fsk\n']:
                submission.write_text('sample_id,predicted_label\n' + text)
                with self.assertRaises(ValueError):
                    score(key, submission)


if __name__ == '__main__':
    unittest.main()
