'''
Unit test for Training Process class in model module.
'''

import os
import unittest
import tempfile
from mufins.common.checkpoint.checkpoint_manager import CheckpointManager
from mufins.tests.common.model.model.mock_model import MockModel
from mufins.tests.common.model.training_process.mock_training_process import (
    MockTrainingProcess, Interruption,
)


#########################################
class TestTrainingProcess(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_interruption(
        self,
    ) -> None:
        '''
        Test that the training process can resume progress after an interruption.
        '''
        checkpoint_id = 'name'
        model = None
        batch_size = 1
        minibatch_size = 2
        num_minibatches = 4
        patience = 10
        max_epochs = 3
        model_path = None
        checkpoint_manager = None

        with tempfile.TemporaryDirectory() as model_path:
            # Test that model does not get modified once it has finished training.

            model = MockModel('sgd')
            checkpoint_manager = CheckpointManager(os.path.join(model_path, 'check.sqlite3'))
            checkpoint_manager.init()

            proc = MockTrainingProcess(expected_first_epoch=1, interrupt_on=None, model=model)
            proc.run(
                checkpoint_id=checkpoint_id, model=model, batch_size=batch_size,
                minibatch_size=minibatch_size, num_minibatches=[num_minibatches],
                patience=patience,
                max_epochs=max_epochs, model_path=model_path,
                checkpoint_manager=checkpoint_manager,
            )
            target_params = [param.data.numpy().tolist() for param in model.net.parameters()]

            proc = MockTrainingProcess(expected_first_epoch=1, interrupt_on=None, model=model)
            proc.run(
                checkpoint_id=checkpoint_id, model=model, batch_size=batch_size,
                minibatch_size=minibatch_size, num_minibatches=[num_minibatches],
                patience=patience,
                max_epochs=max_epochs, model_path=model_path,
                checkpoint_manager=checkpoint_manager,
            )
            self.assertEqual(
                target_params,
                [param.data.numpy().tolist() for param in model.net.parameters()]
            )

        with tempfile.TemporaryDirectory() as model_path:
            # Check that checkpoints resume as expected.

            model = MockModel('sgd')
            checkpoint_manager = CheckpointManager(os.path.join(model_path, 'check.sqlite3'))
            checkpoint_manager.init()

            try:
                proc = MockTrainingProcess(expected_first_epoch=1, interrupt_on=2, model=model)
                proc.run(
                    checkpoint_id=checkpoint_id, model=model, batch_size=batch_size,
                    minibatch_size=minibatch_size, num_minibatches=[num_minibatches],
                    patience=patience,
                    max_epochs=max_epochs, model_path=model_path,
                    checkpoint_manager=checkpoint_manager,
                )
                raise AssertionError('Did not interrupt.')
            except Interruption:
                self.assertNotEqual(
                    target_params,
                    [param.data.numpy().tolist() for param in model.net.parameters()]
                )

            try:
                proc = MockTrainingProcess(expected_first_epoch=2, interrupt_on=3, model=model)
                proc.run(
                    checkpoint_id=checkpoint_id, model=model, batch_size=batch_size,
                    minibatch_size=minibatch_size, num_minibatches=[num_minibatches],
                    patience=patience,
                    max_epochs=max_epochs, model_path=model_path,
                    checkpoint_manager=checkpoint_manager,
                )
                raise AssertionError('Did not interrupt.')
            except Interruption:
                self.assertNotEqual(
                    target_params,
                    [param.data.numpy().tolist() for param in model.net.parameters()]
                )

            proc = MockTrainingProcess(expected_first_epoch=3, interrupt_on=None, model=model)
            proc.run(
                checkpoint_id=checkpoint_id, model=model, batch_size=batch_size,
                minibatch_size=minibatch_size, num_minibatches=[num_minibatches],
                patience=patience,
                max_epochs=max_epochs, model_path=model_path,
                checkpoint_manager=checkpoint_manager,
            )

            out_params = [param.data.numpy().tolist() for param in model.net.parameters()]

        self.assertEqual(target_params, out_params)


#########################################
if __name__ == '__main__':
    unittest.main()
