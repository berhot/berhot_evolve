"""Training spreads across articles, resumes exactly and measures sample loops."""
import copy
import random
import unittest

from evolve.training_schedule import next_window,repetition_metrics,repetition_guard


class TrainingScheduleTests(unittest.TestCase):
    def cursor(self):
        return {'epoch':0,'document_index':0,'byte_offset':0,'attempted_training_bytes':0,'windows_completed':0}

    def test_one_window_per_article_before_revisiting_a_long_article(self):
        documents=[b'a'*80,b'b'*28,b'c'*19]
        cursor=self.cursor();visited=[]
        for _ in documents:
            before=copy.deepcopy(cursor)
            data,next_cursor=next_window(documents,cursor,8)
            self.assertEqual(cursor,before)  # failed optimizer work can retry unchanged
            visited.append(next_cursor['last_document_index'])
            self.assertEqual(len(set(data)),1)  # never crosses article boundaries
            cursor=next_cursor
        self.assertEqual(set(visited),{0,1,2})

    def test_resume_is_deterministic_and_epoch_covers_every_target_byte_once(self):
        documents=[bytes(range(20)),bytes(range(40,67)),bytes(range(100,113))]
        cursor=self.cursor();covered=[0,0,0]
        while True:
            window,planned=next_window(documents,cursor,7)
            other,again=next_window(documents,copy.deepcopy(cursor),7)
            self.assertEqual(window,other);self.assertEqual(planned,again)
            if planned['epoch']>0:break
            covered[planned['last_document_index']]+=len(window)-1;cursor=planned
        self.assertEqual(covered,[len(doc)-1 for doc in documents])

    def test_legacy_cursor_conversion_retains_old_document_offsets(self):
        documents=[b'a'*30,b'b'*40,b'c'*50]
        order=list(range(3));random.Random(42).shuffle(order)
        cursor=self.cursor();cursor.update(document_index=1,byte_offset=8)
        window,planned=next_window(documents,cursor,8)
        offsets=planned['document_offsets']
        self.assertEqual(offsets[order[0]],len(documents[order[0]])-1)
        self.assertEqual(planned['last_document_index'],order[1])
        self.assertEqual(offsets[order[1]],16)

    def test_loop_metric_does_not_treat_known_words_as_fluent_output(self):
        loop=repetition_metrics('في السعودية للملك فهد '*12)
        fresh=repetition_metrics('تأسست الدولة السعودية الأولى في الدرعية وتصف المصادر أحداث نشأتها وتطور الحكم فيها')
        self.assertGreater(loop['repeated_trigram_fraction'],.8)
        self.assertLess(fresh['repeated_trigram_fraction'],.1)
        self.assertFalse(repetition_guard(fresh,loop))
        self.assertTrue(repetition_guard(loop,fresh))


if __name__=='__main__':unittest.main()
