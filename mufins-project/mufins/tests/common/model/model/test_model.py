'''
Unit test for Model class in model module.
'''

import os
import unittest
import tempfile
import torch
import numpy as np
from mufins.tests.common.model.model.mock_model import MockModel


#########################################
class TestModel(unittest.TestCase):
    '''
    Unit test class.
    '''

    #########################################
    def test_gradient(
        self,
    ) -> None:
        '''
        Test that the average gradients are calculated correctly.
        '''
        data = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8]
        ], np.float32)

        model = MockModel('sgd')
        outputs = model.net(data)
        model.opts[0].zero_grad()
        loss = torch.mean((outputs - 1.0)**2)
        loss.backward()
        target_grads = [param.grad.numpy().tolist() for param in model.net.parameters()]

        model = MockModel('sgd')
        model.batch_fit(({'x': data},), 3)
        out_grads = [param.grad.numpy().tolist() for param in model.net.parameters()]

        self.assertEqual(target_grads, out_grads)

    #########################################
    def _test_checkpoint(
        self,
    ) -> None:
        '''
        Test that the saved state will actually continue correctly when loaded.
        '''
        data = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8]
        ], np.float32)

        model = MockModel('adam')
        for _ in range(10):
            model.batch_fit(data, 3)
        target_params = [param.data.numpy().tolist() for param in model.net.parameters()]

        with tempfile.TemporaryDirectory() as path:
            model = MockModel('adam')
            for _ in range(5):
                model.batch_fit(data, 3)
            self.assertNotEqual(
                target_params,
                [param.data.numpy().tolist() for param in model.net.parameters()]
            )
            model.save_state(os.path.join(path, 'x.pkl'))
            model = MockModel('adam')
            model.load_state(os.path.join(path, 'x.pkl'))
            for _ in range(5):
                model.batch_fit(data, 3)
            out_params = [param.data.numpy().tolist() for param in model.net.parameters()]

            self.assertEqual(target_params, out_params)


#########################################
if __name__ == '__main__':
    unittest.main()
